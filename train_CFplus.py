"""Script to train the CF baseline."""

from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import random
import pickle as pkl
from absl import app
from absl import flags
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import nltk

from cf_core.cf_model import CFModel
from cf_core.cf_train import train_batch as train_batch_cf
from cfplus_core.cfplus_eval import run_batch
from cfplus_core import utils_cfplus
from utils import constants
from utils import utils_data
from utils import utils_eval

flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_string(
    "environment_cat", None, "Environment category to train the model on."
)
flags.DEFINE_integer(
    "environment_var", None, "Environment variant (1-4) to train the model on."
)
flags.DEFINE_string("save_tag", None, "Save tag.")
flags.DEFINE_string("log_folder", "./logs", "Folder to save logs.")
# Hyperparameters for the CF model.
flags.DEFINE_integer("hidden_dimension_cf", 3, "Dimension of the learned latent vectors.")
flags.DEFINE_float("lambda_reg_cf", 1e-5, "Weight regularization constant.")
flags.DEFINE_float("learning_rate_cf", 0.01, "Learning rate.")
# Hyperparameters for the FM model.
flags.DEFINE_float("hidden_dimension_fm", 30, "Hidden dimension of the FM model.")
flags.DEFINE_integer("num_iter_fm", 1000, "Number of iterations for FM model.")
flags.DEFINE_float("init_lr_fm", 0.03, "Initial learning rate for FM model.")
flags.DEFINE_float("init_stdev_fm", 0.1, "Initial standard deviation for FM model.")
FLAGS = flags.FLAGS

CONVERGENCE_THRESHOLD = 1e-3
DEVICE = torch.device("cpu")
nltk.download("wordnet")


def calculate_validation_err(
    cf_model: CFModel,
    hyperparams_cf: Dict[str, Union[int, float]],
    hyperparams_fm: Dict[str, Union[int, float]],
    train_batches: Dict[str, Any],
    eval_batches: Dict[str, Any],
    user_list_train: List[str],
    object_combinations: List[Tuple[str, str]],
):
    """Wrapper function to calculate aggregate metrics on validation data."""
    response_generator = run_batch(
        cf_model=cf_model,
        hyperparams_cf=hyperparams_cf,
        hyperparams_fm=hyperparams_fm,
        train_batches=train_batches,
        eval_batches=eval_batches,
        user_list_train=user_list_train,
        object_combinations=object_combinations,
    )
    edit_distance_arr = []
    igo_arr = []
    for i, response in enumerate(response_generator):
        if i % 5 == 0:
            print(f"Validation batch {i}.")
        edit_distance, _, igo = utils_eval.compute_eval_metrics(
            response["predicted_scene"], response["goal_scene"]
        )
        edit_distance_arr.append(edit_distance)
        igo_arr.append(igo)
    return (
        np.mean(edit_distance_arr),
        np.std(edit_distance_arr),
        np.mean(igo_arr),
        np.std(igo_arr),
    )


def train(
    hyperparams_cf: Dict[str, Union[int, float]],
    hyperparams_fm: Dict[str, Union[int, float]],
    train_batches: Dict[str, Dict[str, Any]],
    eval_batches: Dict[str, Dict[str, Any]],
    logfolder: Path,
):
    """Train function."""
    # Load train data for CF.
    matrix_train, user_list_train, non_neg_indices_train, object_combinations = (
        utils_cfplus.batch_to_ranking_matrix(train_batches)
    )
    assert object_combinations is not None
    matrix_train = torch.tensor(
        matrix_train, dtype=torch.double, device=DEVICE, requires_grad=False
    )
    # Initialize model.
    cf_model = CFModel(
        hyperparams_cf["hidden_dimension"],
        len(object_combinations),
        len(user_list_train),
        hyperparams_cf["lambda_reg"],
        mode="train",
    ).double()
    # Train model.
    loss_arr_train = train_batch_cf(
        cf_model,
        matrix_train,
        non_neg_indices_train,
        learning_rate=hyperparams_cf["learning_rate"],
        convergence_threshold=hyperparams_cf["convergence_threshold"],
    )

    # Calculate validation metrics.
    print("Calculating validation metrics...")
    mean_ed, std_ed, mean_igo, std_igo = calculate_validation_err(
        cf_model=cf_model,
        hyperparams_cf=hyperparams_cf,
        hyperparams_fm=hyperparams_fm,
        train_batches=train_batches,
        eval_batches=eval_batches,
        user_list_train=user_list_train,
        object_combinations=object_combinations,
    )
    print(f"Validation Edit Distance: mean {mean_ed:4.2f}, std {std_ed:3.2f}")
    with open(logfolder / "validation_metrics.txt", "w") as fp:
        fp.write(f"edit distance,M{mean_ed:4.2f},S{std_ed:3.2f}\n")
        fp.write(f"igo,M{mean_igo:4.2f},S{std_igo:3.2f}\n")

    # Plot loss.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_arr_train)
    ax.set_title("Train Loss")
    fig.savefig(logfolder / "loss.png")

    # Save model and training parameters.
    torch.save(cf_model.state_dict(), logfolder / "model.ckpt")
    with open(logfolder / "train_params.pkl", "wb") as fp:
        pkl.dump(
            {
                "matrix_train": matrix_train,
                "user_list_train": user_list_train,
                "object_combinations": object_combinations,
            },
            fp,
        )


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    save_tag_template = FLAGS.save_tag
    if save_tag_template is None:
        save_tag_template = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Seed
    np.random.seed(constants.SEED)
    random.seed(constants.SEED)
    torch.random.manual_seed(constants.SEED)

    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError(f"Folder {dataset_folder} does not exist.")

    hyperparams_cf = {
        "hidden_dimension": FLAGS.hidden_dimension_cf,
        "lambda_reg": FLAGS.lambda_reg_cf,
        "learning_rate": FLAGS.learning_rate_cf,
        "convergence_threshold": CONVERGENCE_THRESHOLD,  # TODO: make this a flag.
    }
    hyperparams_fm = {
        "hidden_dimension": FLAGS.hidden_dimension_fm,
        "num_iter": FLAGS.num_iter_fm,
        "init_lr": FLAGS.init_lr_fm,
        "init_stdev": FLAGS.init_stdev_fm,
    }
    print("--Hyperparams--")
    print("CF:")
    for key, value in hyperparams_cf.items():
        print(f"{key}: {value}")
    print("FM:")
    for key, value in hyperparams_fm.items():
        print(f"{key}: {value}")
    print("----\n")
    hyperparams = {
        "cf": hyperparams_cf,
        "fm": hyperparams_fm,
    }

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    logfolder_group = Path(FLAGS.log_folder) / f"cfplus-{save_tag_template}"
    if logfolder_group.exists():
        raise ValueError(f"Group {save_tag_template} exists already.")
    logfolder_group.mkdir(parents=True)

    # Save config file.
    with open(logfolder_group / "config.yaml", "w") as fp:
        yaml.safe_dump(hyperparams, fp)

    # Train on every fold.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Generating process for fold {fkey}.")

        # Create log folder to save checkpoints.
        logfolder = logfolder_group / fkey
        logfolder.mkdir()
        if (logfolder / "model.ckpt").exists():
            raise ValueError(f"Checkpoint exists in {logfolder.name()}.")

        # Combine & filter train and validation data.
        filtered_df = pd.concat(
            [
                fold_df["train"],
                fold_df["val"],
            ],
            ignore_index=True,
        )
        filtered_df = filtered_df[
            filtered_df.num_remaining_partial.eq(0)
            & filtered_df.environment.eq(FLAGS.environment_cat)
            & filtered_df.variant.eq(FLAGS.environment_var)
        ]
        if filtered_df.empty:
            raise ValueError(f"No data for {fkey} with {FLAGS.environment_cat}/{FLAGS.environment_var}.")
        filtered_df = utils_data.return_fold_max_observations(
            filtered_df,
            filtered_df["user_id"].unique().tolist(),
            FLAGS.environment_cat,
            FLAGS.environment_var,
        )

        # Load dataset. Train and validation data are the same.
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            all_batches = {
                k: v
                for k, v in pkl.load(fpt).items()
                if k in filtered_df["scene_id"].to_list()
            }
        with open(dataset_folder / f"{fkey}/val.pickle", "rb") as fpv:
            all_batches.update({
                k: v
                for k, v in pkl.load(fpv).items()
                if k in filtered_df["scene_id"].to_list()
            })
        train_batches = all_batches
        eval_batches = deepcopy(all_batches)

        # Train.
        train(
            hyperparams_cf=hyperparams_cf,
            hyperparams_fm=hyperparams_fm,
            train_batches=train_batches,
            eval_batches=eval_batches,
            logfolder=logfolder,
        )


if __name__ == "__main__":
    app.run(main)
