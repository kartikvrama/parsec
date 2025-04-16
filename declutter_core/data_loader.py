"""Load pre-saved json batches and convert to tensors."""
from typing import List, Dict, Any
import random
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from declutter_core import json_to_tensor
from data_permutation import utils_data_permute
from utils import constants
from utils import data_struct
from utils import utils_data


class DeclutterDataset(Dataset):
    """Data loader to load tensors."""

    data_dict: Dict[str, Any]
    key_list: List[str]

    def __init__(
        self,
        data_dict: List[Any],
        embedding_dict: Dict[str, torch.Tensor],
        grid_dimension: int,
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
        self.batch_generator = json_to_tensor.DeclutterBatchGen(
            object_list,
            surface_constants,
            embedding_dict,
            grid_dimension=grid_dimension
        )

        # Set shuffle flag and load file list.
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
        # utils_data.visualize_scene(goal)
        # utils_data.visualize_scene(new_partial)
        # input("wait")
        return new_partial

    def return_tensor_batch(self, batch_indices: List[int]) -> Dict[str, Any]:
        """Returns a batch of tensors."""
        assert len(batch_indices) == 1
        key = self.key_list[batch_indices[0]]
        batch = {key: self.data_dict[key]}
        if self.mode == "train":
            partial = self.permute_goal(batch[key]["goal"])
            batch[key]["partial"] = partial
        return self.batch_generator.batch_to_tensor(
            batch, is_shuffle=self.is_shuffle
        )
