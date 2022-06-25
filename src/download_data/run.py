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


def main(args):
    """
    Main function of `download_data` step
    """
    run = wandb.init(job_type="download_data")
    run.config.update(args)

    sample_uri = os.path.join(args.sample_url, args.sample)

    # use temp file for logging an artifact to W&B
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Getting data from URI: %s", sample_uri)
        resp = requests.get(sample_uri)

        if not resp.ok:
            resp.raise_for_status()

        tmp_file = os.path.join(temp_dir, 'sample.csv')
        logger.info("Writing sample to temporary file: %s", tmp_file)
        with open(tmp_file, 'w', encoding="utf-8") as temp_file:
            temp_file.write(resp.text)

        artifact = wandb.Artifact(
            args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )

        logger.info("Logging artifact to W&B")
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

    parsed_args = parser.parse_args()

    main(parsed_args)
