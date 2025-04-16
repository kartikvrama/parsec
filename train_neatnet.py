"""Script to train NeatNet model by Kapelyukh et al."""
import os
from datetime import datetime
from pathlib import Path
import pickle as pkl
import random
import yaml
from absl import app
from absl import flags
import pandas as pd

import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor
)

from utils import constants
from utils import utils_data
from utils import utils_torch
from neatnet_core.data_loader import NeatNetDataset
from neatnet_core.neatnet_model import NeatGraph

flags.DEFINE_string(
    "embedding", None, "Path to the object embedding file.",
)
flags.DEFINE_integer("batch_size", 1, "Batch size for training.")
flags.DEFINE_string(
    "user_data_dir",
    None,
    "Path to the folder containing original user arrangements.",
)
flags.DEFINE_string(
    "dataset", None, "Path to the dataset folder."
)
flags.DEFINE_string(
    "fold", None, "Path to the PKL file containing data folds."
)
flags.DEFINE_string(
    "environment_cat", None, "Environment category to train the model on."
)
flags.DEFINE_integer(
    "environment_var", None, "Environment variant (1-4) to train the model on."
)
flags.DEFINE_string("save_tag", None, "Save tag.")
# TODO: create a new script for resuming training.
flags.DEFINE_bool(
    "wandb",
    False,
    "Flag for logging to wandb. Defaults to False",
)
flags.DEFINE_string("log_folder", "./logs", "Folder to save logs.")
# Model Hyperparameters.
flags.DEFINE_integer(
    "graph_dim", 24, "Graph dimension for the transformer model."
)
flags.DEFINE_integer(
    "user_dim", 2, "User dimension for the transformer model."
)
flags.DEFINE_float(
    "relu_leak", 0.2, "Leaky ReLU slope for the transformer model."
)
flags.DEFINE_integer(
    "pos_dim", 2, "Position dimension for the transformer model."
)
flags.DEFINE_integer(
    "semantic_dim", 384, "Object dimension for the transformer model."
)
flags.DEFINE_integer(
    "encoder_h_dim", 24, "Hidden dimension for the encoder model."
)
flags.DEFINE_integer(
    "predictor_h_dim", 32, "Hidden dimension for the predictor model."
)
# Training Hyperparameters.
flags.DEFINE_float(
    "init_lr", 1e-4, "Initial learning rate for the model."
)
flags.DEFINE_integer(
    "num_epochs", 100, "Number of epochs for training."
)
flags.DEFINE_integer(
    "sch_patience", 200, "Patience for the scheduler."
)
flags.DEFINE_integer(
    "sch_cooldown", 100, "Cooldown for the scheduler."
)
flags.DEFINE_float(
    "sch_factor", 0.5, "Factor for the scheduler."
)
flags.DEFINE_float(
    "noise_scale", 0.02, "Noise scale for the model."
)
flags.DEFINE_float(
    "vae_beta", 0.01, "Beta for the VAE loss."
)
FLAGS = flags.FLAGS


def train(
    hyperparams,
    train_batches,
    eval_batches,
    embedding_dict,
    logfolder,
    checkpoint_path,
    wandb_logger,
):
    """Wrapper around train_model function for NeatNet."""
    model = NeatGraph(
        hyperparams=hyperparams, train_mode=True
    ).double()
    print(f"Model\n{model}")

    # Initialize data loaders.
    dataset_train = NeatNetDataset(
        train_batches, embedding_dict, noise_scale=FLAGS.noise_scale, mode="train"
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset_train.return_graph_batch,
    )
    dataset_eval = NeatNetDataset(eval_batches, embedding_dict, mode="eval")
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_eval.return_graph_batch,
    )

    # Save checkpoint for the three highest success rates.
    checkpoint_callback = ModelCheckpoint(
        dirpath=logfolder,
        filename="checkpoint_{epoch:02d}",
        every_n_epochs=500,
        save_top_k=-1,  # <--- this is important!
    )
    val_checkpoint_val_loss = ModelCheckpoint(
        dirpath=logfolder,
        filename="{val_loss:5.3f}-{epoch}-{step}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
    )
    val_checkpoint_mean_edist = ModelCheckpoint(
        dirpath=logfolder,
        filename="{edit_distance:4.2f}-{epoch}-{step}",
        monitor="edit_distance",
        mode="min",
        save_top_k=3,
    )
    val_checkpoint_mean_igo = ModelCheckpoint(
        dirpath=logfolder,
        filename="{igo:4.2f}-{epoch}-{step}",
        monitor="igo",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        checkpoint_callback,
        val_checkpoint_mean_edist,
        val_checkpoint_mean_igo,
        val_checkpoint_val_loss,
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


def split_data_in_distribution(keys_df: pd.DataFrame, val_label: str="C"):
    """Split data into train and validation by excluding goal labels."""
    users = keys_df["user_id"].unique()
    filter_fn = lambda x: any(x.startswith(f"{u}_{val_label}") for u in users)
    val_df = keys_df[keys_df["scene_id"].apply(filter_fn)]  # Predict val_label arrangement in validation.
    train_df = keys_df[~keys_df["scene_id"].isin(val_df["scene_id"])]
    assert not any(train_df["scene_id"].isin(val_df["scene_id"])), "Train and validation keys overlap."
    assert not val_df.empty and not train_df.empty, "Train or validation splits are empty."
    return train_df["scene_id"].to_list(), val_df["scene_id"].to_list()


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
    hyperparams = {
        # Model hyperparameters.
        "graph_dim": FLAGS.graph_dim,
        "user_dim": FLAGS.user_dim,
        "relu_leak": FLAGS.relu_leak,
        "pos_dim": FLAGS.pos_dim,
        "semantic_dim": FLAGS.semantic_dim,
        "encoder_h_dim": FLAGS.encoder_h_dim,
        "predictor_h_dim": FLAGS.predictor_h_dim,
        # Training hyperparameters.
        "init_lr": FLAGS.init_lr,
        "train_batch_size": FLAGS.batch_size,
        "val_batch_size": 1,  #TODO: implement multiple batches for validation.
        "num_epochs": FLAGS.num_epochs,
        "sch_patience": FLAGS.sch_patience,
        "sch_cooldown": FLAGS.sch_cooldown,
        "sch_factor": FLAGS.sch_factor,
        "noise_scale": FLAGS.noise_scale,
        "vae_beta": FLAGS.vae_beta,
    }
    # Update wandb config with hyperparameters.
    wandb_params = hyperparams
    wandb_params.update({
        "environment_cat": FLAGS.environment_cat,
        "environment_var": FLAGS.environment_var,
    })

    print("--Hyperparams--")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("----\n")

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)
    # Load object embeddings.
    embedding_dict = torch.load(FLAGS.embedding)
    for k in embedding_dict.keys():
        embedding_dict[k].to("cuda")

    logfolder_group = Path(FLAGS.log_folder) / f"neatnet-{save_tag_template}"
    if logfolder_group.exists():
        raise ValueError(f"Group {save_tag_template} exists already.")
    logfolder_group.mkdir(parents=True)
    # Save config file.
    with open(logfolder_group / "config.yaml", "w") as fp:
        yaml.safe_dump(wandb_params, fp)

    # Generate multiple training processes.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Generating process for fold {fkey}, env {FLAGS.environment_cat}/{FLAGS.environment_var}.")
        save_tag = f"{save_tag_template}-{fkey}"

        # Initialize wandb session.
        wandb_run, wandb_logger = utils_torch.create_wandb_run(
            project_name="neatnet",
            group_name=save_tag_template,
            run_id=save_tag,
            job_type="train",
            resume=False,
            wandb_params=wandb_params,
        )

        # Create log folder to save checkpoints.
        logfolder = logfolder_group / fkey
        Path(logfolder).mkdir()

        # Load model.
        checkpoint_path = Path(logfolder) / f"neatnet-{save_tag}-final.ckpt"
        if checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_path} exists already.")

        # Filter data based on number of demonstrations.
        filtered_df = pd.concat([fold_df["train"], fold_df["val"]], ignore_index=True)
        filtered_df = filtered_df[
            filtered_df["environment"].eq(FLAGS.environment_cat)
            & filtered_df["variant"].eq(FLAGS.environment_var)
        ]  # TODO: reduce the number of times the dataframe is modified/filtered!
        filtered_df = utils_data.return_fold_max_observations(
            filtered_df,
            filtered_df["user_id"].unique().tolist(),  # TODO: is there a better way to obtain the user list?
            FLAGS.user_data_dir,
            environment_cat=FLAGS.environment_cat,
            environment_var=FLAGS.environment_var,
        )
        all_keys = filtered_df["scene_id"].to_list()
        if all_keys is None:
            raise ValueError(f"No demonstrations found for {FLAGS.environment_cat}/{FLAGS.environment_var} in {fkey}.")
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            all_batches = {k: v for k, v in pkl.load(fpt).items() if k in all_keys}
        with open(dataset_folder / f"{fkey}/val.pickle", "rb") as fpv:
            all_batches.update({k: v for k, v in pkl.load(fpv).items() if k in all_keys})

        # Split data into train and validation by excluding label C from training.
        train_keys, val_keys = split_data_in_distribution(filtered_df, val_label="C")
        assert sorted(train_keys + val_keys) == sorted(all_keys), "Train and validation keys are missing some keys.."
        train_batches = {k: v for k, v in all_batches.items() if k in train_keys}
        eval_batches = {k: v for k, v in all_batches.items() if k in val_keys}
        print(f"Number of training keys: {len(train_batches)} and validation keys: {len(eval_batches)}.")

        train(
            hyperparams,
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
