"""Script to train NeatNet model by Kapelyukh et al."""
from datetime import datetime
from pathlib import Path
import pickle as pkl
import csv
import json
import yaml
import random
from absl import app
from absl import flags
import torch
import numpy as np

from utils import constants
from utils import utils_data
from utils import utils_eval
from neatnet_core.data_loader import NeatNetDataset
from neatnet_core import neatnet_eval

flags.DEFINE_string(
    "embedding", None, "Path to the object embedding file.",
)
flags.DEFINE_string(
    "dataset", None, "Path to the dataset folder."
)
flags.DEFINE_string(
    "fold", None, "Path to the PKL file containing data folds."
)
flags.DEFINE_string(
    "model_tag", None, "Model tag to save results."
)
flags.DEFINE_string(
    "environment_cat", None, "Environment category to train the model on."
)
flags.DEFINE_integer(
    "environment_var", None, "Environment variant (1-4) to train the model on."
)
flags.DEFINE_string(
    "checkpoint_folder", None, "Path to the checkpoint folder for all folds."
)
flags.DEFINE_enum(
    "stopping_metric",
    "edit_distance",
    ["loss", "edit_distance", "igo"],
    "Criteria for early stopping."
)
FLAGS = flags.FLAGS

np.random.seed(constants.SEED)
random.seed(constants.SEED)
torch.random.manual_seed(constants.SEED)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    date_tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_tag = FLAGS.model_tag
    if model_tag is None:
        raise ValueError("Please specify model tag.")

    checkpoint_folder = Path(FLAGS.checkpoint_folder)
    if not checkpoint_folder.exists():
        raise ValueError("Please specify a valid checkpoint folder.")

    if not (checkpoint_folder / "config.yaml").exists():
        print(f"WARNING: config.yaml not found in {checkpoint_folder}. Using default config.")
        with open("configs/neatnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(checkpoint_folder / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

    if FLAGS.dataset is None:
        raise ValueError("Please specify dataset folder.")
    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError("Please specify dataset folder.")

    # Dictionary of model and test parameters.
    hyperparams = {
        # Model hyperparameters.
        "graph_dim": config["graph_dim"],
        "user_dim": config["user_dim"],
        "relu_leak": config["relu_leak"],
        "pos_dim": config["pos_dim"],
        "semantic_dim": config["semantic_dim"],
        "encoder_h_dim": config["encoder_h_dim"],
        "predictor_h_dim": config["predictor_h_dim"],
        # Training hyperparameters.
        "init_lr": config["init_lr"],
        "num_epochs": config["num_epochs"],
        "sch_patience": config["sch_patience"],
        "sch_cooldown": config["sch_cooldown"],
        "sch_factor": config["sch_factor"],
        "noise_scale": config["noise_scale"],
        "vae_beta": config["vae_beta"],
    }
    print("--Hyperparams--")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("----\n")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/neatnet-{model_tag}/{FLAGS.environment_cat}/{FLAGS.environment_var}")
    if result_folder_parent.exists():
        print(f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder.")
        result_folder_parent = Path(f"results/neatnet-{FLAGS.model_tag}-{date_tag}/{FLAGS.environment_cat}/{FLAGS.environment_var}")
    result_folder_parent.mkdir(parents=True)

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Load constants and embeddings.
    embedding_dict = torch.load(FLAGS.embedding)
    for k in embedding_dict.keys():
        embedding_dict[k].to("cuda")

    # Generate results for multiple folds.
    checkpoint_path_dict = {}
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Evaluating on fold {fkey}, env {FLAGS.environment_cat}/{FLAGS.environment_var}.")

        # Filter test data based on environment category, variant and num observations.
        filtered_df = fold_df["test"][
            fold_df["test"]["environment"].eq(FLAGS.environment_cat)
            & fold_df["test"]["variant"].eq(FLAGS.environment_var)
        ]  # TODO: This might be redundant.
        try:
            fold_filtered = utils_data.return_fold_max_observations(
                filtered_df,
                user_list=filtered_df["user_id"].unique().tolist(),
                environment_cat=FLAGS.environment_cat,
                environment_var=FLAGS.environment_var,
            )
        except ValueError as e:
            print(f"Skipping fold {fkey}, issue: {e}.")
            continue

        if fold_filtered.empty:
            raise ValueError(f"{fkey}: no evaluation data.")
        key_list = fold_filtered["scene_id"].tolist()
        if key_list is None:
            raise ValueError(f"No demonstrations found for {FLAGS.environment_cat}/{FLAGS.environment_var} in {fkey}.")
        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items() if key in key_list
            }
        batch_gen_test = NeatNetDataset(test_batches, embedding_dict, mode="eval")

        # Load model.
        try:
            checkpoint_name, _ = utils_eval.return_early_stopping_ckpt(
                checkpoint_folder / fkey, FLAGS.stopping_metric, is_min=True
            )
        except FileNotFoundError as e:
            print(f"Skipping fold {fkey}, issue: {e}.")
            continue

        checkpoint_path = checkpoint_folder / fkey / checkpoint_name
        checkpoint_path_dict[fkey] = str(checkpoint_path.absolute())
        if not checkpoint_path.exists():
            raise ValueError(f"{checkpoint_name} does not exist in {checkpoint_folder / fkey}.")

        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        result_array, prediction_dict, latent_embedding_dict = neatnet_eval.evaluate(
            hyperparams=hyperparams,
            batch_generator=batch_gen_test,
            checkpoint_path=checkpoint_path,
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
        "checkpoint_folder": str(checkpoint_folder.absolute()),
        "checkpoint_dict": checkpoint_path_dict,
    }
    with open(result_folder_parent / "evaluation_metadata.json", "w") as fjson:
        json.dump(eval_metadata, fjson, indent=4, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
