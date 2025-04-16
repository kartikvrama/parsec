"""Script to evaluate Abdo-CF baseline."""
from datetime import datetime
from pathlib import Path
import csv
import json
import pickle as pkl
from absl import app
from absl import flags
import yaml
from nltk.corpus import wordnet as wn

from cf_core import cf_eval
from utils import utils_data
from utils import utils_wn

flags.DEFINE_string(
    "dataset",
    None,
    "Path to the dataset folder."
)
flags.DEFINE_string(
    "fold", None, "Path to the PKL file containing data folds."
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
        print(f"WARNING: config.yaml not found in {checkpoint_folder_parent}. Using default config.")
        with open("configs/cf_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(checkpoint_folder_parent / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

    if FLAGS.dataset is None:
        raise ValueError("Please specify dataset folder.")
    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError("Please specify dataset folder.")

    hyperparams = {
        "hidden_dimension": config["hidden_dimension"],
        "lambda_reg": config["lambda_reg"],
        "learning_rate": config["learning_rate"],
        "convergence_threshold": config["convergence_threshold"],
    }
    print("--Hyperparams--")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("----\n")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/cf-{model_tag}")
    if result_folder_parent.exists():
        print(f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder.")
        result_folder_parent = Path(f"results/cf-{FLAGS.model_tag}-{date_tag}")
    result_folder_parent.mkdir(parents=True)

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

    # Generate results for multiple folds.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Evaluating on fold {fkey}.")

        # Filter test data based on environment category, variant and num observations.
        filtered_df_test = utils_data.return_fold_max_observations(
            fold_df["test"],
            fold_df["test"]["user_id"].unique().tolist(),
        )
        if filtered_df_test.empty:
            raise ValueError(f"Empty test data for {fkey}.")
        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items()
                if key in filtered_df_test["scene_id"].tolist()
            }
        
        checkpoint_folder = checkpoint_folder_parent / fkey
        print(f"Loading checkpoint from {checkpoint_folder}.")
        if not (checkpoint_folder / "model.ckpt").exists() or (checkpoint_folder / "train_params.ckpt").exists():
            raise ValueError(f"Missing files in {checkpoint_folder.name}.")

        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        result_array, prediction_dict, latent_embedding_dict = cf_eval.evaluate(
            hyperparams=hyperparams,
            test_batches=test_batches,
            checkpoint_folder=checkpoint_folder,
            object_ID_list_semantic=object_ID_list_semantic,
            semantic_sim_mat=semantic_sim_mat,
            object_id_dict=object_id_dict,
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
    }
    with open(result_folder_parent / "evaluation_metadata.json", "w") as fjson:
        json.dump(eval_metadata, fjson, indent=4, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
