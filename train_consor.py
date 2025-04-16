"""Script to train ConSOR."""
import os
from datetime import datetime
from pathlib import Path
import pickle as pkl
import random
import yaml
from absl import app
from absl import flags
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor
)

from utils import constants
from utils import utils_data
from utils import utils_torch
from consor_core.consor_model import ConSORTransformer
from consor_core.data_loader import ConSORDataset

flags.DEFINE_string(
    "embedding", None, "Path to the object embedding file.",
)
flags.DEFINE_string(
    "dataset", None, "Path to the dataset folder."
)
flags.DEFINE_string(
    "fold", None, "Path to the PKL file containing data folds."
)
flags.DEFINE_string("save_tag", None, "Save tag.")
#TODO: separate script for resuming training.
flags.DEFINE_bool(
    "wandb",
    False,
    "Flag for logging to wandb. Defaults to False",
)
flags.DEFINE_string("log_folder", "./logs", "Folder to save logs.")
# Model Hyperparameters.
flags.DEFINE_integer("semantic_embb_dim", 384, "Semantic embedding dimension.")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer dimension.")
flags.DEFINE_integer("output_dimension", 256, "Output dimension.")
flags.DEFINE_integer("num_heads", 4, "Number of encoder attention heads.")
flags.DEFINE_integer("num_layers", 2, "Number of transformer encoder layers.")
flags.DEFINE_float("dropout", 0.1, "Dropout rate.")
flags.DEFINE_integer("object_pos_encoding_dim", 32, "Object positional encoding dimension.")
flags.DEFINE_integer(
    "surface_pos_encoding_dim", 32, "Surface positional encoding dimension."
)
# Training Hyperparameters.
flags.DEFINE_integer("num_epochs", 5, "Number of epochs.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_float("lrate", 1e-3, "Learning rate.")
flags.DEFINE_float("wt_decay", 1e-20, "Weight decay.")
flags.DEFINE_enum(
    "loss_fn", "triplet_margin", ["triplet_margin", "npairs"], "Metric loss function."
)
flags.DEFINE_float("triplet_loss_margin", None, "Triplet loss margin.")

FLAGS = flags.FLAGS
torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_start_method("spawn", force=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(
    hyperparams,
    train_batches,
    eval_batches,
    embedding_dict,
    logfolder,
    checkpoint_path,
    wandb_logger,
):
    """Wrapper around train_model function for ConSOR."""
    model = ConSORTransformer(**hyperparams).double()
    print(f"Model\n{model}")

    # Initialize data loaders.
    dataset_train = ConSORDataset(
        train_batches,
        embedding_dict,
        object_pos_encoding_dim=FLAGS.object_pos_encoding_dim,
        surface_pos_encoding_dim=FLAGS.surface_pos_encoding_dim,
        mode="train"
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=hyperparams["train_batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=dataset_train.return_tensor_batch,
    )
    dataset_eval = ConSORDataset(
        eval_batches,
        embedding_dict,
        object_pos_encoding_dim=FLAGS.object_pos_encoding_dim,
        surface_pos_encoding_dim=FLAGS.surface_pos_encoding_dim,
        mode="eval",
    )
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=1,  # TODO: add a flag for val batch size.
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_eval.return_tensor_batch,
    )

    # Save checkpoint for the three highest success rates.
    checkpoint_callback = ModelCheckpoint(
        dirpath=logfolder,
        filename="checkpoint_{epoch:02d}",
        every_n_epochs=250,
        save_top_k=-1,  # <--- this is important!
    )
    val_checkpoint_sr = ModelCheckpoint(
        dirpath=logfolder,
        filename="{success_rate:4.2f}-{epoch}-{step}",
        monitor="success_rate",
        mode="max",
        save_top_k=3,
    )
    val_checkpoint_edist = ModelCheckpoint(
        dirpath=logfolder,
        filename="{edit_distance:4.2f}-{epoch}-{step}",
        monitor="edit_distance",
        mode="min",
        save_top_k=3,
    )
    val_checkpoint_igo = ModelCheckpoint(
        dirpath=logfolder,
        filename="{igo:4.2f}-{epoch}-{step}",
        monitor="igo",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        checkpoint_callback,
        val_checkpoint_sr,
        val_checkpoint_edist,
        val_checkpoint_igo,
        TQDMProgressBar(refresh_rate=500),
        lr_monitor,
    ]

    utils_torch.train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_eval=dataloader_eval,
        logfolder=logfolder,
        num_epochs=FLAGS.num_epochs,
        wandb_logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_path=checkpoint_path,
    )


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "dryrun"

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

    # Dictionary of model and training parameters.
    data_params = {
        "semantic_embb_dim": FLAGS.semantic_embb_dim,
        "object_pos_encoding_dim": FLAGS.object_pos_encoding_dim,
        "surface_pos_encoding_dim": FLAGS.surface_pos_encoding_dim
    }
    model_params = {
        "input_dim": (
            FLAGS.semantic_embb_dim + FLAGS.object_pos_encoding_dim
            + FLAGS.surface_pos_encoding_dim
        ),
        "hidden_layer_size": FLAGS.hidden_layer_size,
        "output_dimension": FLAGS.output_dimension,
        "num_heads": FLAGS.num_heads,
        "num_layers": FLAGS.num_layers,
        "dropout": FLAGS.dropout,
        "train_batch_size": FLAGS.batch_size,
        "val_batch_size": 1,  # TODO: add a flag for val batch size.
        "lrate": FLAGS.lrate,
        "wt_decay": FLAGS.wt_decay,
        "loss_fn": FLAGS.loss_fn,
        "triplet_loss_margin": FLAGS.triplet_loss_margin,
    }

    # Create wandb config from model and data parameters.
    wandb_params = {}
    wandb_params.update(model_params)
    wandb_params.update(data_params)
    wandb_params.update({"num_epochs": FLAGS.num_epochs})

    print("--Data params--")
    for key, value in data_params.items():
        print(f"{key}: {value}")
    print("----\n")
    print("--Model params--")
    for key, value in model_params.items():
        print(f"{key}: {value}")
    print("----\n")

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)
    # Load object embeddings.
    embedding_dict = torch.load(FLAGS.embedding)
    for k in embedding_dict.keys():
        embedding_dict[k].to("cuda")
    if not embedding_dict["alexa"].shape[0] == FLAGS.semantic_embb_dim:
        raise ValueError(
            f"Mismatch in semantic embedding dimension: {embedding_dict['alexa'].shape[0]} vs {FLAGS.semantic_embb_dim}."
        )
    embedding_dict.update(
        {
            constants.EMPTY_LABEL: torch.zeros(
                (FLAGS.semantic_embb_dim,), dtype=torch.float64, device="cuda"
            )
        }
    )
    # Create log folder to save checkpoints.
    logfolder_group = Path(FLAGS.log_folder) / f"consor-{save_tag_template}"
    if logfolder_group.exists():
        raise ValueError(f"Group {save_tag_template} exists already.")
    logfolder_group.mkdir(parents=True)
    # Save config file.
    with open(logfolder_group / "config.yaml", "w") as fp:
        yaml.safe_dump(wandb_params, fp)

    # Train on all folds.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Generating process for fold {fkey}.")
        save_tag = f"{save_tag_template}-{fkey}"

        # Initialize wandb session.
        wandb_run, wandb_logger = utils_torch.create_wandb_run(
            project_name="consor",
            group_name=save_tag_template,
            run_id=save_tag,
            job_type="train",
            resume=False,
            wandb_params=wandb_params,
        )

        # Create log folder to save checkpoints.
        logfolder = logfolder_group / fkey
        Path(logfolder).mkdir()
        checkpoint_path = Path(logfolder) / f"consor-{save_tag}-final.ckpt"
        if checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_path} exists already.")

        # Filter keys based on number of demonstrations.
        fold_df_train = utils_data.return_filtered_fold(
            fold_df["train"], num_demonstations=0
        )
        if fold_df_train is None:
            raise ValueError(f"No train keys with 0 demonstrations in {fkey}.")
        fold_df_val = utils_data.return_filtered_fold(
            fold_df["val"], num_demonstations=0
        )
        if fold_df_val is None:
            raise ValueError(f"No val keys with 0 demonstrations in {fkey}.")

        # Load dataset.
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            train_batches = {
                k: v for k, v in pkl.load(fpt).items()
                if k in fold_df_train["scene_id"].to_list()
            }
        with open(dataset_folder / f"{fkey}/val.pickle", "rb") as fpv:
            eval_batches = {
                k: v for k, v in pkl.load(fpv).items()
                if k in fold_df_val["scene_id"].to_list()
            }

        # Train model.
        train(
            model_params,
            train_batches,
            eval_batches,
            embedding_dict,
            logfolder,
            checkpoint_path,
            wandb_logger,
        )
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)
