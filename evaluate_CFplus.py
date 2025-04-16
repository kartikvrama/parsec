"""Script to evaluate Abdo-CF baseline."""
from datetime import datetime
from pathlib import Path
import csv
import json
import pickle as pkl
from absl import app
from absl import flags
import yaml
import pandas as pd

from utils import utils_data
from cfplus_core import cfplus_eval

flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_string(
    "environment_cat", None, "Environment category to train the model on."
)
flags.DEFINE_integer(
    "environment_var", None, "Environment variant (1-4) to train the model on."
)
flags.DEFINE_string(
    "model_tag", None, "Model tag to save results."
)
flags.DEFINE_string(
    "checkpoint_folder", None, "Path to the checkpoint folder for all folds."
)
FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    date_tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_tag = FLAGS.model_tag
    if model_tag is None:
        raise ValueError("Please specify model tag.")

    checkpoint_folder_parent = Path(FLAGS.checkpoint_folder)
    if not checkpoint_folder_parent.exists():
        raise ValueError("Please specify a valid checkpoint folder.")

    if not (checkpoint_folder_parent / "config.yaml").exists():
        raise NotImplementedError("Default config not implemented.")
    else:
        with open(checkpoint_folder_parent / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

    hyperparams_cf = {
        "hidden_dimension": config["cf"]["hidden_dimension"],
        "lambda_reg": config["cf"]["lambda_reg"],
        "learning_rate": config["cf"]["learning_rate"],
        "convergence_threshold": config["cf"]["convergence_threshold"],
    }
    hyperparams_fm = {
        "hidden_dimension": config["fm"]["hidden_dimension"],
        "num_iter": config["fm"]["num_iter"],
        "init_lr": config["fm"]["init_lr"],
        "init_stdev": config["fm"]["init_stdev"],
    }
    print("--Hyperparams--")
    print("CF:")
    for key, value in hyperparams_cf.items():
        print(f"{key}: {value}")
    print("FM:")
    for key, value in hyperparams_fm.items():
        print(f"{key}: {value}")
    print("----\n")

    if FLAGS.dataset is None:
        raise ValueError("Please specify dataset folder.")
    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError("Please specify dataset folder.")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/cfplus-{model_tag}/{FLAGS.environment_cat}/{FLAGS.environment_var}")
    if result_folder_parent.exists():
        print(f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder.")
        result_folder_parent = Path(f"results/neatnet-{FLAGS.model_tag}-{date_tag}/{FLAGS.environment_cat}/{FLAGS.environment_var}")
    result_folder_parent.mkdir(parents=True)

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Generate results for multiple folds.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Evaluating on fold {fkey}, env {FLAGS.environment_cat}/{FLAGS.environment_var}.")

        # Load test data.
        filtered_df_test = fold_df["test"][
            fold_df["test"].environment.eq(FLAGS.environment_cat)
            & fold_df["test"].variant.eq(FLAGS.environment_var)
        ]
        try:
            filtered_df_test = utils_data.return_fold_max_observations(
                filtered_df_test,
                filtered_df_test["user_id"].unique().tolist(),
                FLAGS.environment_cat,
                FLAGS.environment_var,
            )
        except ValueError as e:
            print(f"Skipping fold {fkey}, issue: {e}.")
            continue
        if filtered_df_test.empty:
            raise ValueError(f"Empty test data for {FLAGS.environment_cat}/{FLAGS.environment_var} in {fkey}.")
        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items()
                if key in filtered_df_test["scene_id"].tolist()
            }

        # Load train data for the FM model.
        filtered_df_train = pd.concat(
            [
                fold_df["train"],
                fold_df["val"],
            ],
            ignore_index=True,
        )
        filtered_df_train = filtered_df_train[
            filtered_df_train.num_remaining_partial.eq(0)
            & filtered_df_train.environment.eq(FLAGS.environment_cat)
            & filtered_df_train.variant.eq(FLAGS.environment_var)
        ]
        filtered_df_train = utils_data.return_fold_max_observations(
            filtered_df_train,
            filtered_df_train["user_id"].unique().tolist(),
            FLAGS.environment_cat,
            FLAGS.environment_var,
        )
        if filtered_df_train.empty:
            raise ValueError(f"No data for {fkey}.")
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            train_batches = {
                k: v
                for k, v in pkl.load(fpt).items()
                if k in filtered_df_train["scene_id"].to_list()
            }

        checkpoint_folder = checkpoint_folder_parent / fkey
        print(f"Loading checkpoint from {checkpoint_folder}.")
        if not (checkpoint_folder / "model.ckpt").exists() or (checkpoint_folder / "train_params.ckpt").exists():
            raise ValueError(f"Missing files in {checkpoint_folder.name}.")

        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        result_array, prediction_dict, latent_embedding_dict = cfplus_eval.evaluate(
            checkpoint_folder=checkpoint_folder,
            hyperparams_cf=hyperparams_cf,
            hyperparams_fm=hyperparams_fm,
            train_batches=train_batches,
            eval_batches=test_batches,
        )
        # Save results.
        with open(result_folder / "results.csv", "w") as fcsv:
            csv_writer = csv.DictWriter(fcsv, fieldnames=result_array[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(result_array)
        with open(result_folder / "predictions.pkl", "wb") as fp:
            pkl.dump(prediction_dict, fp)
        with open(result_folder / "latent_embeddings.pkl", "wb") as fp:
            pkl.dump(latent_embedding_dict, fp)

    # Save metadata.
    eval_metadata = {
        "dataset_folder": str(dataset_folder.absolute()),
        "checkpoint_folder": str(checkpoint_folder_parent.absolute()),
    }
    with open(result_folder_parent / "evaluation_metadata.json", "w") as fjson:
        json.dump(eval_metadata, fjson, indent=4, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
