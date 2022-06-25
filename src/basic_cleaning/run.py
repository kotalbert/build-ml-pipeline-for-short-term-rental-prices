#!/usr/bin/env python
"""
Basic cleaning of the raw data and logging the artifact to W&B
"""
import argparse
import logging
import os
from tempfile import TemporaryDirectory

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading Artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info("Reading data")
    df = pd.read_csv(artifact_path)

    logger.info("Remove `price` outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting `last_review` to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Preparing data Artifact for logging with W&B")
    with TemporaryDirectory() as td:
        filename = os.path.join(td, args.output_artifact)
        df.to_csv(filename)

        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description
        )
        artifact.add_file(filename)

        logger.info("Logging Artifact to W&B")
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output Artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum house price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum house price",
        required=True
    )

    args = parser.parse_args()

    go(args)
