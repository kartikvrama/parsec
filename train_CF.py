"""Script to train the CF baseline."""

from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import random
import pickle as pkl
from absl import app
from absl import flags
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn

from cf_core import cf_train
from cf_core.cf_model import CFModel
from cf_core import cf_eval
from cf_core import utils_cf
from utils import constants
from utils import utils_data
from utils import utils_eval
from utils import utils_wn

flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_string("save_tag", None, "Save tag.")
flags.DEFINE_string("log_folder", "./logs", "Folder to save logs.")
flags.DEFINE_integer("hidden_dimension", 3, "Dimension of the learned latent vectors.")
flags.DEFINE_float("lambda_reg", 1e-5, "Weight regularization constant.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
FLAGS = flags.FLAGS

CONVERGENCE_THRESHOLD = 1e-3
DEVICE = torch.device("cpu")
nltk.download("wordnet")


def calculate_validation_err(
    model: CFModel,
    hyperparams: Dict[str, Union[int, float]],
    eval_batches: Dict[str, Any],
    user_list_train: List[str],
    object_combinations: List[Tuple[str, str]],
    object_ID_list_semantic: List[str],
    semantic_similarity_matrix: np.ndarray,
    object_id_dict: Dict[str, int],
):
    """Wrapper function to calculate aggregate metrics on validation data."""
    response_generator = cf_eval.run_batch(
        model=model,
        hyperparams=hyperparams,
        eval_batches=eval_batches,
        user_list_train=user_list_train,
        object_combinations=object_combinations,
        object_ID_list_semantic=object_ID_list_semantic,
        semantic_similarity_matrix=semantic_similarity_matrix,
        object_id_dict=object_id_dict,
    )
    edit_distance_arr = []
    igo_arr = []
    for i, response in enumerate(response_generator):
        if i % 2 == 0:
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
    hyperparams: Dict[str, Union[int, float]],
    train_batches: Dict[str, Dict[str, Any]],
    eval_batches: Dict[str, Dict[str, Any]],
    object_ID_list_semantic: List[str],
    semantic_similarity_matrix: np.ndarray,
    object_id_dict: Dict[str, int],
    logfolder: Path,
):
    """Train function."""
    # Load train data.
    matrix_train, user_list_train, non_neg_indices_train, object_combinations = (
        utils_cf.batch_to_ranking_matrix(train_batches)
    )
    assert object_combinations is not None
    matrix_train = torch.tensor(
        matrix_train, dtype=torch.double, device=DEVICE, requires_grad=False
    )
    # Initialize model.
    model = CFModel(
        hyperparams["hidden_dimension"],
        len(object_combinations),
        len(user_list_train),
        hyperparams["lambda_reg"],
        mode="train",
    ).double()
    # Train model.
    loss_arr_train = cf_train.train_batch(
        model,
        matrix_train,
        non_neg_indices_train,
        learning_rate=hyperparams["learning_rate"],
        convergence_threshold=hyperparams["convergence_threshold"],
    )

    # Calculate validation metrics.
    print("Calculating validation metrics...")
    mean_ed, std_ed, mean_igo, std_igo = calculate_validation_err(
        model=model,
        hyperparams=hyperparams,
        eval_batches=eval_batches,
        user_list_train=user_list_train,
        object_combinations=object_combinations,
        object_ID_list_semantic=object_ID_list_semantic,
        semantic_similarity_matrix=semantic_similarity_matrix,
        object_id_dict=object_id_dict,
    )
    print(f"Validation IGO: mean {mean_igo:4.2f}, std {std_igo:3.2f}")
    with open(logfolder / "validation_metrics.txt", "w") as fp:
        fp.write(f"edit distance,M{mean_ed:4.2f},S{std_ed:3.2f}\n")
        fp.write(f"igo,M{mean_igo:4.2f},S{std_igo:3.2f}\n")

    # Plot loss.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_arr_train)
    ax.set_title("Train Loss")
    fig.savefig(logfolder / "loss.png")

    # Save model and training parameters.
    torch.save(model.state_dict(), logfolder / "model.ckpt")
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

    hyperparams = {
        "hidden_dimension": FLAGS.hidden_dimension,
        "lambda_reg": FLAGS.lambda_reg,
        "learning_rate": FLAGS.learning_rate,
        "convergence_threshold": CONVERGENCE_THRESHOLD,  # TODO: make this a flag.
    }
    print("--Hyperparams--")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("----\n")

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Create a semantic similarity matrix using wordnet labels.
    object_id_dict, _ = utils_data.return_object_surface_constants()
    object_ID_list_semantic = list(object_id_dict.keys())  # List of object ids.
    obj_wn_label_dict = utils_wn.load_wordnet_labels()
    synsets = [
        wn.synset(obj_wn_label_dict[object_id_dict[oid]["name"]])
        for oid in object_ID_list_semantic
    ]
    semantic_sim_mat = utils_wn.return_pairwise_distance_matrix(synsets)

    logfolder_group = Path(FLAGS.log_folder) / f"cf-{save_tag_template}"
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

        # Filter data.
        filtered_df_train = utils_data.return_fold_max_observations(
            fold_df["train"],
            fold_df["train"]["user_id"].unique().tolist(),
        )
        filtered_df_val = utils_data.return_fold_max_observations(
            fold_df["val"],
            fold_df["val"]["user_id"].unique().tolist(),
        )
        # Only consider empty scenes for validation.
        filtered_df_val = filtered_df_val[filtered_df_val.num_remaining_partial.eq(0)]
        if filtered_df_train.empty:
            raise ValueError(f"Empty training data for {fkey}.")
        if filtered_df_val.empty:
            raise ValueError(f"Empty validation data for {fkey}.")

        # Load dataset.
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            train_batches = {
                k: v
                for k, v in pkl.load(fpt).items()
                if k in filtered_df_train["scene_id"].to_list()
            }
        with open(dataset_folder / f"{fkey}/val.pickle", "rb") as fpv:
            eval_batches = {
                k: v
                for k, v in pkl.load(fpv).items()
                if k in filtered_df_val["scene_id"].to_list()
            }

        # Train.
        train(
            hyperparams=hyperparams,
            train_batches=train_batches,
            eval_batches=eval_batches,
            object_ID_list_semantic=object_ID_list_semantic,
            semantic_similarity_matrix=semantic_sim_mat,
            object_id_dict=object_id_dict,
            logfolder=logfolder,
        )


if __name__ == "__main__":
    app.run(main)
