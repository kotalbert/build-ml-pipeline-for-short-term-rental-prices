#!/usr/bin/env python
"""
Training, evaluation and logging to W&B of the Random Forest Inference Artifact
"""
import argparse
import json
import logging
import os
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
import wandb
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    rf_config['random_state'] = args.random_seed

    logger.info("Fetching artifact: %s", args.train_artifact)
    train_local_path = run.use_artifact(args.train_artifact).file()

    x = pd.read_csv(train_local_path)
    y = x.pop("price")

    logger.info("Minimum price: %f, Maximum price: %f", y.min(), y.max())

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_size, stratify=x[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    logger.info("Fitting")
    sk_pipe.fit(x_train, y_train)

    logger.info("Scoring")
    r_squared = sk_pipe.score(x_val, y_val)

    y_pred = sk_pipe.predict(x_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info("Score: %f", r_squared)
    logger.info("MAE: %f", mae)

    logger.info("Exporting model")
    with TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, args.output_artifact)
        model_signature = infer_signature(x_val[processed_features], y_pred)
        mlflow.sklearn.save_model(
            sk_pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=model_signature,
            input_example=x_val[processed_features].iloc[:5])

        artifact = wandb.Artifact(
            args.output_artifact,
            type="model_export",
            description="Random Forest Inference Artifact"
        )
        artifact.add_dir(export_path)
        run.log_artifact(artifact)
        artifact.wait()

    logger.info("Create feature importance plots")
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    run.summary['r2'] = r_squared
    run.summary['mea'] = mae

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe: Pipeline, feat_names: List[str]):
    """
    Plot feature importance
    """
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names) - 1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config: Dict, max_tfidf_features: int) -> Tuple[Pipeline, List[str]]:
    """
    Build sklearn inference pipeline, based on hyperparameters from provided config dictionary

    The following operations are included:
        * Encoding categorical variables,
        * Missing value imputations,
        * Fitting
    """
    ordinal_categorical = ["room_type"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical = ["neighbourhood_group"]
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    random_forest_classifier = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest_classifier)])

    return sk_pipe, processed_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--train_artifact",
        type=str,
        help="A fully qualified name of Artifact containing the training dataset. "
             "It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
             "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output inference artifact",
        required=True,
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
