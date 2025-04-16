"""Load pre-saved json batches and convert to graphs."""
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
import random
import torch
from torch.utils.data import Dataset
from neatnet_core import json_to_graph
from utils import utils_data

PERMUTE_PROB = 0.2
NOISE_SCALE = 0.05


class NeatNetDataset(Dataset):
    """Data loader to load graphs."""

    data_dict: Dict[str, Any]
    key_list: List[str]

    def __init__(
        self,
        data_dict: List[Any],
        embedding_dict: Dict[str, torch.Tensor],
        noise_scale: Optional[float] = None,
        mode: str = "train",
    ):
        """Initializes data loader class and loads JSON data to memory."""

        # Load object labels, surface constants, and RoBERTa embeddings.
        obj_label_dict, surface_constants = utils_data.return_object_surface_constants()
        object_list = list(d["text"] for d in obj_label_dict.values())
        if any(o not in embedding_dict for o in object_list):
            raise ValueError("Missing objects in the embedding dictionary.")

        # Initialize batch generator.
        self.batch_generator = json_to_graph.NeatNetBatchGen(
            object_list, surface_constants, embedding_dict
        )

        # Set shuffle flag and load file list.
        self.noise_scale = noise_scale
        if noise_scale is None:
            self.noise_scale = NOISE_SCALE
        self.mode = mode
        self.is_shuffle = mode == "train"
        self.data_dict = data_dict
        self.key_list = list(self.data_dict.keys())
        print(f"Number of batches: {len(data_dict)}")
        self.surface_constants = surface_constants

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, pos_id):
        return pos_id

    def permute_scenes(self, scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly omit observations scenes from the input."""
        new_scene_dict = deepcopy(scene_dict)
        ommited_obs_scenes = []
        for scene in new_scene_dict["observed_arrangement_list"]:
            if random.random() < PERMUTE_PROB:
                continue
            ommited_obs_scenes.append(scene)
        if len(ommited_obs_scenes) == 0:
            random_scene = random.choice(new_scene_dict["observed_arrangement_list"])
            ommited_obs_scenes.append(random_scene)
        new_scene_dict["observed_arrangement_list"] = ommited_obs_scenes
        return new_scene_dict

    def add_pos_noise(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Adds noise to the example node positions."""
        new_true_positions = batch["train_pred_positions"] + self.noise_scale * torch.randn_like(
            batch["train_pred_positions"]
        )
        batch["train_pred_positions"] = new_true_positions
        return batch

    def return_graph_batch(
        self, batch_indices: Union[int, List[int]]
    ) -> Dict[str, Any]:
        """Returns a batch of graphs."""
        if isinstance(batch_indices, int):
            batch_indices = [batch_indices]
        batch_keys = list(self.key_list[b] for b in batch_indices)
        if self.mode == "train":
            batch = {k: self.permute_scenes(self.data_dict[k]) for k in batch_keys}
        else:
            batch = {k: self.data_dict[k] for k in batch_keys}
        batch = self.batch_generator.batch_to_tensor(batch, is_shuffle=self.is_shuffle)
        if self.mode == "train":
            return self.add_pos_noise(batch)
        return batch
