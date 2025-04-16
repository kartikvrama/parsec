"""Load JSON examples and convert to tensors for training and evaluation."""
from typing import List, Dict, Any
import random
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from consor_core import json_to_tensor
from data_permutation import utils_data_permute
from utils import constants
from utils import data_struct
from utils import utils_data


class ConSORDataset(Dataset):
    """Data loader to load tensors."""

    data_dict: Dict[str, Any]
    key_list: List[str]

    def __init__(
        self, data_dict: List[Any], embedding_dict: Dict[str, torch.Tensor],
        object_pos_encoding_dim: int = 4, surface_pos_encoding_dim: int = 8,
        mode: str = "train"
    ):
        """Initializes data loader class and loads JSON data to memory."""

        # Load object labels, surface constants, and RoBERTa embeddings.
        obj_label_dict, surface_constants = utils_data.return_object_surface_constants()
        object_list = list(d["text"] for d in obj_label_dict.values())
        object_list = [constants.EMPTY_LABEL] + object_list
        if any(o not in embedding_dict for o in object_list):
            raise ValueError("Missing objects in the embedding dictionary.")

        # Initialize batch generator.
        self.batch_generator = json_to_tensor.ConSORBatchGen(
            object_list, surface_constants, embedding_dict,
            object_pos_encoding_dim=object_pos_encoding_dim,
            surface_pos_encoding_dim=surface_pos_encoding_dim
        )

        # Set class variables.
        self.mode = mode
        self.data_dict = data_dict
        self.key_list = list(self.data_dict.keys())
        print(f"Number of batches: {len(data_dict)}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, pos_id):
        return pos_id

    def permute_goal(
        self, goal: data_struct.SurfaceEntity
    ) -> data_struct.SurfaceEntity:
        """Remove objects from goal and place on unplaced surface."""
        new_partial = [deepcopy(s) for s in goal]
        objects_to_remove = []
        for surface in new_partial:
            objects_to_remove.extend(list(
                o.object_id for o in surface.objects_on_surface
                if random.random() < 0.5
            ))
        for obj_id in objects_to_remove:
            utils_data.remove_object_from_scene(
                new_partial, obj_id, move_to_unplaced=True
            )
        return new_partial

    def return_tensor_batch(self, batch_indices: List[int]) -> Dict[str, Any]:
        """Returns a batch of tensors."""
        key_list = list(self.key_list[i] for i in batch_indices)
        batch = {
            k: self.data_dict[k] for k in key_list
        }
        if self.mode == "train":
            for key in key_list:
                partial = self.permute_goal(batch[key]["goal"])
                batch[key]["partial"] = partial
        return self.batch_generator.batch_to_tensor(
            batch, shuffle=(self.mode == "train")
        )
