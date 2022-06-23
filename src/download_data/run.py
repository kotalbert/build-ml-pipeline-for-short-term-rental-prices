#!/usr/bin/env python
"""
Download data and create a W&B Artifact to be used in pipeline
"""
import argparse
import logging
import os
import tempfile

import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="download_data")
    run.config.update(args)

    sample_uri = os.path.join(args.sample_url, args.sample)

    # use temp file for logging an artifact to W&B
    with tempfile.TemporaryDirectory() as td:
        logger.info(f"Getting data from URI: {sample_uri}")
        r = requests.get(sample_uri)

        if not r.ok:
            r.raise_for_status()
        tmp_file = os.path.join(td, 'sample.csv')
        logger.info(f"Writing sample to temporary file: {tmp_file}")
        with open(tmp_file, 'w') as tf:
            tf.write(r.text)

        artifact = wandb.Artifact(
            args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )

        artifact.add_file(tmp_file)
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data and create W&B Artifact")

    parser.add_argument("sample_url", type=str, help="URL, where data for download is")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)
