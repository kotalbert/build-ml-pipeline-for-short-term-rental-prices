import json
import os

import hydra
import mlflow
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps, so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    #    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config', config_path=".")
def go(config: DictConfig):
    # Set up the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config.main.project_name
    os.environ["WANDB_RUN_GROUP"] = config.main.experiment_name

    # Steps to execute
    steps_par = config.main.steps
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    def get_step_abs_pth(step_name: str) -> str:
        """
        Get absolute path to step to run it with mlflow.
        Throw error if path does not exist.
        """
        step_abs_pth = os.path.join(hydra.utils.get_original_cwd(), "src", step_name)
        assert os.path.isdir(step_abs_pth)
        return step_abs_pth

    # put data name literal to variable, reuse it in multiple steps
    raw_data_filename = "sample.csv"

    if "download" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            get_step_abs_pth("download_data"),
            "main",
            parameters={
                "sample_url": config.etl.sample_url,
                "sample": config.etl.sample,
                "artifact_name": raw_data_filename,
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded"
            },
        )

    clean_data_filename = "clean_sample.csv"
    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            get_step_abs_pth("basic_cleaning"),
            "main",
            parameters={
                "input_artifact": f"{raw_data_filename}:latest",
                "output_artifact": clean_data_filename,
                "output_type": "clean_sample",
                "output_description": "Data after basic cleaning",
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price
            }
        )

    if "data_check" in active_steps:
        _ = mlflow.run(
            get_step_abs_pth("data_check"),
            "main",
            parameters={
                "csv": f"{clean_data_filename}:latest",
                "ref": f"{clean_data_filename}:reference",
                "kl_threshold": config.data_check.kl_threshold,
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price
            }
        )

    if "data_split" in active_steps:
        _ = mlflow.run(
            get_step_abs_pth("data_split"),
            "main",
            parameters={
                "input_artifact": f"{clean_data_filename}:latest",
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by

            }
        )

    if "train_random_forest" in active_steps:
        # NOTE: we need to serialize the random forest configuration into JSON
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

        # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
        # step

        ##################
        # Implement here #
        ##################

        pass

    if "test_regression_model" in active_steps:
        ##################
        # Implement here #
        ##################

        pass


if __name__ == "__main__":
    go()
