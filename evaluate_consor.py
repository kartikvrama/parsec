"""Script to train ConSOR model by Kapelyukh et al."""
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
from consor_core import consor_eval
from consor_core.json_to_tensor import ConSORBatchGen

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
    "checkpoint_folder", None, "Path to the checkpoint folder for all folds."
)
flags.DEFINE_enum(
    "stopping_metric",
    "edit_distance",
    ["success_rate", "edit_distance", "igo"],
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
        with open("configs/consor_config.yaml", "r") as f:
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
    data_params = {
        "semantic_embb_dim": config["semantic_embb_dim"],
        "object_pos_encoding_dim": config["object_pos_encoding_dim"],
        "surface_pos_encoding_dim": config["surface_pos_encoding_dim"]
    }
    model_params = {
        "input_dim": config["input_dim"],
        "hidden_layer_size": config["hidden_layer_size"],
        "output_dimension": config["output_dimension"],
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        #TODO: keep train and validation batch sizes separate.
        "train_batch_size": config["train_batch_size"],
        "val_batch_size": config["val_batch_size"],
        "lrate": config["lrate"],
        "wt_decay": config["wt_decay"],
        "loss_fn": config["loss_fn"],
        "triplet_loss_margin": config["triplet_loss_margin"],
    }
    print("--Data params--")
    for key, value in data_params.items():
        print(f"{key}: {value}")
    print("----\n")
    print("--Model params--")
    for key, value in model_params.items():
        print(f"{key}: {value}")
    print("----\n")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/consor-{model_tag}")
    if result_folder_parent.exists():
        print(f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder.")
        result_folder_parent = Path(f"results/{FLAGS.model_tag}-{date_tag}")
    result_folder_parent.mkdir(parents=True)

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Load object embeddings.
    embedding_dict = torch.load(FLAGS.embedding)
    for k in embedding_dict.keys():
        embedding_dict[k].to("cuda")
    if not embedding_dict["alexa"].shape[0] == config["semantic_embb_dim"]:
        raise ValueError(
            f"Mismatch in semantic embedding dimension: {embedding_dict['alexa'].shape[0]} vs {config['semantic_embb_dim']}."
        )
    embedding_dict.update(
        {
            constants.EMPTY_LABEL: torch.zeros(
                (config["semantic_embb_dim"],), dtype=torch.float64, device="cuda"
            )
        }
    )

    # Populate object list.
    obj_label_dict, surface_constants = utils_data.return_object_surface_constants()
    object_list = list(d["text"] for d in obj_label_dict.values())
    object_list = [constants.EMPTY_LABEL] + object_list

    # Initialize batch generator.
    batch_gen_test = ConSORBatchGen(
        object_list,
        surface_constants,
        embedding_dict,
        object_pos_encoding_dim=data_params["object_pos_encoding_dim"],
        surface_pos_encoding_dim=data_params["surface_pos_encoding_dim"],
    )

    # Generate results for multiple folds.
    checkpoint_path_dict = {}
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Evaluating on fold {fkey}.")

        # Filter test data based on environment category, variant and num observations.
        fold_df_test = fold_df["test"][fold_df["test"]["num_demonstrations"].eq(0)]
        if fold_df_test.empty:
            raise ValueError(f"No keys with 0 demonstrations found for {fkey}.")
        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items()
                if key in fold_df_test["scene_id"].tolist()
            }

        # Load model.
        if FLAGS.stopping_metric == "success_rate":
            checkpoint_name, _ = utils_eval.return_early_stopping_ckpt(
                checkpoint_folder / fkey, "success_rate", is_min=False
            )
        else:
            checkpoint_name, _ = utils_eval.return_early_stopping_ckpt(
                checkpoint_folder / fkey, FLAGS.stopping_metric, is_min=True
            )
        checkpoint_path = checkpoint_folder / fkey / checkpoint_name
        checkpoint_path_dict[fkey] = str(checkpoint_path.absolute())
        if not checkpoint_path.exists():
            raise ValueError(f"{checkpoint_name} does not exist in {checkpoint_folder / fkey}.")
        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        result_array, prediction_dict, latent_embedding_dict = consor_eval.evaluate(
            hyperparams=model_params,
            checkpoint_path=checkpoint_path,
            batch_list=test_batches,
            batch_generator=batch_gen_test,
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
