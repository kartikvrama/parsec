"""Helper functions for PyTorch."""
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from wandb.errors import CommError

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_wandb_run(
    project_name: str,
    group_name: str,
    run_id: str,
    job_type: str="train",
    resume: bool=False,
    wandb_params: Dict[str, Union[str, int, float]]=None
):
    """Creates a unique wandb run for a given project, group, and run name.
    
    Args:
        project_name: Name of the wandb project.
        group_name: Name of the wandb group.
        run_id: Name of the wandb run.
        job_type: Type of job (e.g., train, eval).
        resume: Whether to resume the wandb run.
        wandb_params: Dictionary of wandb parameters.
    """
    run_id = f"{run_id}-{wandb.util.generate_id()}"

    run = wandb.init(
        project=project_name,
        group=group_name,
        id=run_id,
        settings=wandb.Settings(code_dir="."),
        job_type=job_type,
        resume=resume,
    )
    if wandb_params is not None:
        run.config.update(wandb_params)

    wandb_logger = WandbLogger(
        name=f"log_{run_id}", project=project_name, log_model=False
    )
    return run, wandb_logger


def train_model(
    model: pl.LightningModule,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    logfolder: Union[str, Path],
    num_epochs: int,
    wandb_logger: WandbLogger,
    callbacks: List[pl.Callback],
    checkpoint_path: Union[str, Path],
    profiler: Optional[str]=None,
    detect_anomaly: bool=False
):
    """Initializes lightning trainer and trains the model.
    
    Refer to pytorch lightning documentation on Trainer for more information
    about arguments.

    Args:
        model: Pytorch lightning model.
        dataloader_train: Training dataloader.
        dataloader_eval: Evaluation dataloader.
        logfolder: Path to save logs.
        num_epochs: Number of epochs to train.
        wandb_logger: Wandb logger.
        callbacks: List of callbacks.
        checkpoint_path: Path to save model checkpoint.
        profiler: Profiler to use (e.g., "simple").
        detect_anomaly: Whether to detect anomalies.
    """

    # Pytorch lighning trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=num_epochs,
        default_root_dir=logfolder,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler=profiler,
        detect_anomaly=detect_anomaly
    )

    trainer.validate(model, dataloader_eval)
    trainer.fit(model, dataloader_train, dataloader_eval)
    trainer.save_checkpoint(checkpoint_path)


def return_positional_encoding_dict(max_position, d_model, min_freq=1e-4):
    """Returns a sinusoidal positional encoding generator.

    Args:
        max_position: Maximum number of positions.
        d_model: Positional encoding representation.
        min_freq: Minimum frequency of the sinusoidal positional encoding.

    Returns:
        An array of supported positions and a tensor of positional embeddings.
    """
    position = torch.arange(max_position, device=DEVICE)
    freqs = min_freq ** (
        2 * (torch.arange(d_model, device=DEVICE) // 2) / d_model
    )
    pos_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)
    # Apply cosine to even indices and sine to odd indices.
    pos_enc[:, ::2] = torch.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = torch.sin(pos_enc[:, 1::2])
    return position, pos_enc


class PositionalEncodingCaller:
    """Class to generate positional encodings from indices."""
    pos_encoding_mat: torch.Tensor

    def __init__(self, max_position, d_model, min_freq=1e-4):
        """Initializes the positional encoding caller."""
        _, self.pos_encoding_mat = return_positional_encoding_dict(
            max_position, d_model, min_freq=min_freq
        )

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns positional encodings for a list of indices."""
        if len(indices.size()) > 1:
            return torch.stack(
                [self.pos_encoding_mat[i] for i in indices]
            )
        elif len(indices.size()) == 1 and len(indices) == 1:
            return self.pos_encoding_mat[indices].unsqueeze(0)
        return self.pos_encoding_mat[indices]


class EmbeddingCaller:
    """Class to generate embeddings from labels."""
    embedding_dict: Dict[Union[int, str, torch.Tensor], torch.Tensor]
    object_dimension: int

    def __init__(self, label_list, embedding_dict=None):
        """Initializes the embedding caller with a list of labels and embeddings."""
        if embedding_dict is not None:
            assert all(lb in embedding_dict for lb in label_list)
            self.embedding_matrix = torch.stack(
                [embedding_dict[lb].ravel() for lb in label_list]
            ).to(torch.double)
        else:
            self.embedding_matrix = torch.arange(
                len(label_list), dtype=torch.long, device=DEVICE
            ).reshape(-1, 1)
        self.label_list = label_list
        self.object_dimension = self.embedding_matrix.size()[-1]

    def __call__(self, label) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(label, torch.Tensor):
            raise NotImplementedError("Tensor labels not supported.")
        if isinstance(label, list):
            indices = torch.tensor([self.label_list.index(l) for l in label], device=DEVICE)
            return torch.index_select(self.embedding_matrix, 0, indices)
        index = torch.tensor(self.label_list.index(label), device=DEVICE)
        return torch.index_select(self.embedding_matrix, 0, index).view(1, -1)

    def get_dimension(self):
        """Returns the dimension of the embeddings."""
        return self.object_dimension


def merge_scene_graphs(scene_nodes: torch.tensor, scene_edges: torch.tensor):
    """Merges independent scene graphs into a single graph.
    
    Copied from NeatNet implementation by Kapelyukh et al.
    """
    combined_nodes = torch.vstack(scene_nodes)
    num_scenes = len(scene_nodes)
    num_nodes = combined_nodes.size(0)
    scene_ids = torch.empty(num_nodes, dtype=torch.long, device=DEVICE)

    num_edges = 0
    for edges in scene_edges:
        num_edges += edges.size(1)

    combined_edges = torch.empty(2, num_edges, dtype=torch.long, device=DEVICE)
    nodes_by_scene = torch.empty((num_scenes,), dtype=torch.long, device=DEVICE)
    nodes_so_far = 0
    edges_so_far = 0
    for scene_idx in range(num_scenes):
        nodes_in_scene = scene_nodes[scene_idx]
        edges_in_scene = scene_edges[scene_idx]

        new_edges_in_scene = edges_in_scene + nodes_so_far
        combined_edges[:, edges_so_far : edges_so_far + new_edges_in_scene.size(1)] = edges_in_scene
        scene_ids[nodes_so_far : nodes_so_far + nodes_in_scene.size(0)] = scene_idx

        nodes_so_far += nodes_in_scene.size(0)
        edges_so_far += new_edges_in_scene.size(1)
        nodes_by_scene[scene_idx] = nodes_in_scene.size(0)

    return combined_nodes, combined_edges, scene_ids, nodes_by_scene


def pad_tensor_from_list(tensor_list: List[torch.tensor]):
    """Return a padded list of tensors with a mask and batch split.

    The function pads a list of tensors to fit the maximum length tensor in the
    list and creates a padding mask of zeros and ones, with ones corresponding
    to the indices with padding. The function also returns a list of tensor
    lengths before padding to reconstruct the true tensor.

    Args:
        tensor_list: A list (B, ) of tensors with unequal lengths.

    Returns:
        Paddded tensor of dimension (B, N_{padded}, d_{final}),
        Padding mask of dimension (B, N_{padded}),
        List of tensor lenghts (B,)
    """
    assert all(len(x.size()) == 2 for x in tensor_list)
    tensor_final_dim = tensor_list[0].size()[1:]
    if not all(x.size()[1:] == tensor_final_dim for x in tensor_list):
        raise ValueError("Tensors have different final dimensions.")

    padded_tensor = pad_sequence(tensor_list, batch_first=True).to(torch.double)
    tensor_split = [x.shape[0] for x in tensor_list]
    max_length = max(tensor_split)
    padding_mask = torch.concat(
        [
            torch.concat(
                [
                    torch.zeros([1, x.shape[0]]),
                    torch.ones([1, max_length - x.shape[0]]),
                ],
                dim=1,
            ).to(tensor_list[0].get_device())
            for x in tensor_list
        ], dim=0
    ).to(torch.double)

    return padded_tensor, padding_mask, tensor_split
