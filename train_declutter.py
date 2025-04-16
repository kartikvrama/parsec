"""Script to train Declutter."""
import os
from pathlib import Path
from datetime import datetime
import pickle as pkl
import random
import yaml
from absl import app
from absl import flags
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
from declutter_core.data_loader import DeclutterDataset
from declutter_core.encoder_decoder_model import DeclutterEncoderDecoder

flags.DEFINE_string(
    "embedding", None, "Path to the object embedding file.",
)
flags.DEFINE_string(
    "dataset", None, "Path to the dataset for training.",
)
flags.DEFINE_string(
    "fold", None, "Path to the PKL file containing data folds."
)
flags.DEFINE_string(
    "save_tag", None, "Save tag."
)
flags.DEFINE_bool(
    "wandb", False, "Flag for logging to wandb. Defaults to False",
)
flags.DEFINE_string(
    "log_folder", "./logs", "Folder to save logs."
)
# Model Hyperparameters.
flags.DEFINE_integer(
    "num_heads", 1, "Number of heads for the transformer model."
)
flags.DEFINE_integer(
    "num_layers", 1, "Number of layers for the transformer model."
)
flags.DEFINE_integer(
    "hidden_layer_size", 1024, "Hidden layer size for the transformer model."
)
flags.DEFINE_float(
    "dropout", 0.5, "Dropout for the transformer model."
)
flags.DEFINE_integer(
    "object_dimension", 384, "Object dimension for the transformer model."
)
flags.DEFINE_integer(
    "num_container_types", 5, "Number of container types."
)
flags.DEFINE_integer(
    "num_surface_types", 6, "Number of surface types."
)
flags.DEFINE_integer(
    "surface_grid_dimension", 64, "Surface grid dimension."
)
flags.DEFINE_integer(
    "type_embedding_dim", 1, "Type embedding dimension."
)
flags.DEFINE_integer(
    "instance_encoder_dim", 256, "Instance encoder dimension."
)
# Training Hyperparameters.
flags.DEFINE_integer(
    "num_epochs", 10, "Number of epochs for training."
)
flags.DEFINE_integer(
    "batch_size", 1, "Batch size for training."
)
flags.DEFINE_float(
    "lrate", 1e-5, "Learning rate for training."
)
flags.DEFINE_float(
    "wt_decay", 1e-20, "Weight decay for training."
)
flags.DEFINE_integer(
    "lr_scheduler_tmax", -1, "Learning rate scheduler tmax."
)
flags.DEFINE_float(
    "alpha", 1.0, "Alpha for triplet loss."
)
flags.DEFINE_float(
    "beta", 0.5, "Beta for triplet loss."
)
flags.DEFINE_float(
    "triplet_margin_main", 0.75, "Triplet margin for main loss."
)
flags.DEFINE_float(
    "triplet_margin_aux", 0.75, "Triplet margin for auxiliary loss."
)

FLAGS = flags.FLAGS
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train(
    model_params,
    train_params,
    train_batches,
    eval_batches,
    embedding_dict,
    logfolder,
    checkpoint_path,
    wandb_logger,

):
    """Wrapper around train_model function for Declutter."""
    model = DeclutterEncoderDecoder(
        model_params=model_params,
        batch_size=float(train_params["batch_size"]),
        lrate=float(train_params["lrate"]),
        wt_decay=float(train_params["wt_decay"]),
        lr_scheduler_tmax = train_params["lr_scheduler_tmax"],
        alpha = float(train_params["alpha"]),
        beta = float(train_params["beta"]),
        triplet_margin_main = float(train_params["triplet_margin_main"]),
        triplet_margin_aux = float(train_params["triplet_margin_aux"]),
        mode="train",
    ).double()
    print(f"Model\n{model}")

    # Initialize train and validation data loaders.
    # TODO: Add support for multiple batches.
    dataset_train = DeclutterDataset(
        train_batches,
        embedding_dict,
        grid_dimension=model_params["surface_grid_dimension"],
        mode="train"
    )
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=0,
        batch_size=1,
        collate_fn=dataset_train.return_tensor_batch,
        shuffle=True,
    )
    dataset_eval = DeclutterDataset(
        eval_batches,
        embedding_dict,
        grid_dimension=model_params["surface_grid_dimension"],
        mode="val"
    )
    dataloader_eval = DataLoader(
        dataset_eval,
        num_workers=0,
        batch_size=1,
        collate_fn=dataset_eval.return_tensor_batch,
        shuffle=False,
    )

    # Save checkpoint for the three highest success rates.
    checkpoint_callback = ModelCheckpoint(
        dirpath=logfolder,
        filename="checkpoint_{epoch:02d}",
        every_n_epochs=4,
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
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [
        checkpoint_callback,
        val_checkpoint_sr,
        val_checkpoint_edist,
        val_checkpoint_igo,
        TQDMProgressBar(refresh_rate=150),
        lr_monitor,
    ]

    utils_torch.train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_eval=dataloader_eval,
        logfolder=logfolder,
        num_epochs=int(train_params["num_epochs"]),
        wandb_logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_path=checkpoint_path,
    )


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    date_tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if FLAGS.lr_scheduler_tmax == -1:
        lr_scheduler_tmax = "None"
    else:
        lr_scheduler_tmax = FLAGS.lr_scheduler_tmax

    if FLAGS.wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "dryrun"

    # Seed
    np.random.seed(constants.SEED)
    random.seed(constants.SEED)
    torch.random.manual_seed(constants.SEED)

    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError(f"Folder {dataset_folder} does not exist.")

    save_tag_template = FLAGS.save_tag
    if save_tag_template is None:
        save_tag_template = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Dictionary of model and training parameters.
    model_params = {
        "num_heads": FLAGS.num_heads,
        "num_layers": FLAGS.num_layers,
        "hidden_layer_size": FLAGS.hidden_layer_size,
        "dropout": FLAGS.dropout,
        "object_dimension": FLAGS.object_dimension,
        "num_container_types": FLAGS.num_container_types,
        "num_surface_types": FLAGS.num_surface_types,
        "surface_grid_dimension": FLAGS.surface_grid_dimension,
        "type_embedding_dim": FLAGS.type_embedding_dim,
        "instance_encoder_dim": FLAGS.instance_encoder_dim,
    }
    train_params = {
        "dataset_folder": str(dataset_folder.absolute()),
        "num_epochs": FLAGS.num_epochs,
        "batch_size": FLAGS.batch_size,
        "lrate": FLAGS.lrate,
        "wt_decay": FLAGS.wt_decay,
        "lr_scheduler_tmax": lr_scheduler_tmax,
        "alpha": FLAGS.alpha,
        "beta": FLAGS.beta,
        "triplet_margin_main": FLAGS.triplet_margin_main,
        "triplet_margin_aux": FLAGS.triplet_margin_aux,
        "save_tag": save_tag_template
    }

    wandb_params = {}
    wandb_params.update(model_params)
    wandb_params.update(train_params)

    print("--Train params--")
    for key, value in train_params.items():
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
    if not embedding_dict["alexa"].shape[0] == FLAGS.object_dimension:
        raise ValueError(
            f"Mismatch in semantic embedding dimension: {embedding_dict['alexa'].shape[0]} vs {FLAGS.object_dimension}."
        )
    embedding_dict.update(
        {
            constants.EMPTY_LABEL: torch.zeros(
                (FLAGS.object_dimension,), dtype=torch.float64, device="cuda"
            )
        }
    )

    # Create log folder to save checkpoints.
    logfolder_group = Path(FLAGS.log_folder) / f"declutter-{save_tag_template}"
    if logfolder_group.exists():
        print(f"Group {save_tag_template} exists already, resuming training.")
    else:
        logfolder_group.mkdir(parents=True)
    # Save config file.
    with open(logfolder_group / "config.yaml", "w") as fp:
        yaml.safe_dump(wandb_params, fp)

    # Train on all folds.
    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue

        if (logfolder_group / fkey).exists():
            print(f"Folder for {fkey} exists already.")
            if (logfolder_group / fkey / f"declutter-{save_tag_template}-{fkey}-final.ckpt").exists():
                print(f"Checkpoint for fold {fkey} exists already.")
                continue
            else:
                print(f"Fold {fkey} exists but training was not complete, creating new temporary folder.")
                save_tag = f"{save_tag_template}-{fkey}-{date_tag}"
                logfolder = logfolder_group / f"{fkey}-{date_tag}"
        else:
            print(f"Training on fold {fkey}.")
            save_tag = f"{save_tag_template}-{fkey}"
            logfolder = logfolder_group / fkey

        # Initialize wandb session.
        wandb_run, wandb_logger = utils_torch.create_wandb_run(
            project_name="declutter",
            group_name=save_tag_template,
            run_id=save_tag,
            job_type="train",
            resume=False,
            wandb_params=wandb_params,
        )

        # Create log folder to save checkpoints.
        Path(logfolder).mkdir()
        checkpoint_path = Path(logfolder) / f"declutter-{save_tag}-final.ckpt"
        if checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_path} exists already.")

        # Filter data based on number of demonstrations.
        filtered_df_train = utils_data.return_fold_max_observations(
            fold_df["train"],
            fold_df["train"]["user_id"].unique().tolist(),  # TODO: is there a better way to obtain the user list?
        )
        if filtered_df_train.empty:
            raise ValueError(f"Empty training data for {fkey}")
        filtered_df_val = utils_data.return_fold_max_observations(
            fold_df["val"],
            fold_df["val"]["user_id"].unique().tolist(),
        )
        if filtered_df_val.empty:
            raise ValueError(f"Empty validation data for {fkey}")

        # Load dataset.
        with open(dataset_folder / f"{fkey}/train.pickle", "rb") as fpt:
            train_batches = {
                k: v for k, v in pkl.load(fpt).items()
                if k in filtered_df_train["scene_id"].to_list()
            }
        with open(dataset_folder / f"{fkey}/val.pickle", "rb") as fpv:
            eval_batches = {
                k: v for k, v in pkl.load(fpv).items()
                if k in filtered_df_val["scene_id"].to_list()
            }
        
        # Train model.
        train(
            model_params,
            train_params,
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
