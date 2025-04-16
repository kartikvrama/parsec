"""Script to evaluate Declutter."""

import csv
import json
from datetime import datetime
from pathlib import Path
import pickle as pkl

import random
import yaml
from absl import app
from absl import flags
import numpy as np
import torch

from utils import constants
from utils import utils_data
from utils import utils_eval
from declutter_core import json_to_tensor
from declutter_core import declutter_eval

flags.DEFINE_string(
    "embedding",
    None,
    "Path to the object embedding file.",
)
flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_string("model_tag", None, "Model tag to save results.")
flags.DEFINE_string(
    "checkpoint_folder", None, "Path to the checkpoint folder for all folds."
)
flags.DEFINE_enum(
    "stopping_metric",
    "edit_distance",
    ["success_rate", "edit_distance", "igo"],
    "Criteria for early stopping.",
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
        raise ValueError("Please specify checkpoint folder.")

    if not (checkpoint_folder / "config.yaml").exists():
        print(
            f"WARNING: config.yaml not found in {checkpoint_folder}. Using default config."
        )
        with open("configs/declutter_config.yaml", "r") as f:
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
    model_params = {
        "num_heads": int(config["num_heads"]),
        "num_layers": int(config["num_layers"]),
        "hidden_layer_size": int(config["hidden_layer_size"]),
        "dropout": float(config["dropout"]),
        "object_dimension": int(config["object_dimension"]),
        "num_container_types": int(config["num_container_types"]),
        "num_surface_types": int(config["num_surface_types"]),
        "surface_grid_dimension": int(config["surface_grid_dimension"]),
        "type_embedding_dim": int(config["type_embedding_dim"]),
        "instance_encoder_dim": int(config["instance_encoder_dim"]),
    }
    print("--Model params--")
    for key, value in model_params.items():
        print(f"{key}: {value}")
    print("----\n")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/declutter-{model_tag}")
    if result_folder_parent.exists():
        print(
            f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder."
        )
        result_folder_parent = Path(f"results/{FLAGS.model_tag}-{date_tag}")
    result_folder_parent.mkdir(parents=True)

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Load object embeddings.
    embedding_dict = torch.load(FLAGS.embedding)
    for k in embedding_dict.keys():
        embedding_dict[k].to("cuda")
    if not embedding_dict["alexa"].shape[0] == config["object_dimension"]:
        raise ValueError(
            f"Mismatch in semantic embedding dimension: {embedding_dict['alexa'].shape[0]} vs {config['object_dimension']}."
        )
    embedding_dict.update(
        {
            constants.EMPTY_LABEL: torch.zeros(
                (config["object_dimension"],), dtype=torch.float64, device="cuda"
            )
        }
    )

    # Initialize batch generator.
    obj_label_dict, surface_constants = utils_data.return_object_surface_constants()
    object_list = list(d["text"] for d in obj_label_dict.values()) + [
        constants.EMPTY_LABEL
    ]
    batch_generator = json_to_tensor.DeclutterBatchGen(
        object_list=object_list,
        surface_constants=surface_constants,
        object_embedding_dict=embedding_dict,
        grid_dimension=int(config["surface_grid_dimension"]),
    )

    # Generate results for multiple folds.
    checkpoint_path_dict = {}
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Evaluating on {fkey}.")

        fold_filtered = utils_data.return_fold_max_observations(
            fold_df["test"],
            user_list=fold_df["test"]["user_id"].unique().tolist(),  # TODO: is there a better way to obtain the user list?
        )
        key_list = fold_filtered["scene_id"].tolist()
        if key_list is None:
            raise ValueError(f"Empty test dataset for {fkey}.")
        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items() if key in key_list
            }

        checkpoint_name, _ = utils_eval.return_early_stopping_ckpt(
            checkpoint_folder / fkey, FLAGS.stopping_metric, is_min=True
        )
        checkpoint_path = checkpoint_folder / fkey / checkpoint_name
        checkpoint_path_dict[fkey] = str(checkpoint_path.absolute())
        if not checkpoint_path.exists():
            raise ValueError(
                f"{checkpoint_name} does not exist in {checkpoint_folder / fkey}."
            )

        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        result_array, prediction_dict, latent_embedding_dict = declutter_eval.evaluate(
            model_params=model_params,
            checkpoint_path=checkpoint_path,
            batch_dict=test_batches,
            batch_generator=batch_generator,
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
