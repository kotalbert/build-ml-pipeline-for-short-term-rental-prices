#!/usr/bin/env python
"""
This script splits the provided dataframe to train and test sets
"""
import argparse
import logging
import os
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(args):
    """
    Main function of data_split step
    """
    run = wandb.init(job_type="data_split")
    run.config.update(args)

    logger.info("Fetching artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting sample to training and test sets")
    train, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    for df, set_type in zip([train, test], ['train', 'test']):
        artifact_name = f"{set_type}_data.csv"
        logger.info("Writing to temporary file: %s", artifact_name)

        with TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, artifact_name)
            df.to_csv(temp_file, index=False)

            artifact = wandb.Artifact(
                artifact_name,
                type=set_type,
                description=f"{set_type.capitalize()} data for model fitting"
            )

            logger.info("Loading artifact to W&B")
            artifact.add_file(temp_file)
            run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data to train and test samples")

    parser.add_argument("--input_artifact", type=str, help="Fully qualified name of the input artifact to split")

    parser.add_argument("--test_size", type=float, help="Size of the test split. Fraction of the dataset, or number "
                                                        "of items")

    parser.add_argument("--random_seed", type=int, help="Seed for random number generator", default=42, required=False)

    parser.add_argument("--stratify_by", type=str, help="Column to use for stratification", default='none',
                        required=False)

    parsed_args = parser.parse_args()

    main(parsed_args)
