"""Class to convert JSON data to tensors for training and evaluating ConSOR."""

from typing import Any, Dict, List, Literal, Optional, Tuple
import random
import numpy as np
import torch

from utils import utils_data
from utils import utils_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConSORBatchGen:
    """Transforms JSON examples into tensor batches."""

    def __init__(
        self,
        object_list: List[str],
        surface_constants: Dict[str, Dict[str, str]],
        object_embedding_dict: Optional[Dict[str, torch.Tensor]],
        object_pos_encoding_dim: int=4,
        surface_pos_encoding_dim: int=8,
    ):
        """Initialize embedding dictionaries and positional encodings."""
        self.surface_constants = surface_constants
        self.object_name_to_tensor = utils_torch.EmbeddingCaller(
            object_list, object_embedding_dict
        )
        self.position_encoder_object = utils_torch.PositionalEncodingCaller(
            max_position=40, d_model=object_pos_encoding_dim
        )
        self.position_encoder_surface = utils_torch.PositionalEncodingCaller(
            max_position=15, d_model=surface_pos_encoding_dim
        )

    def batch_to_tensor(
        self, batch: Dict[str, Any], shuffle: bool = True
    ) -> Dict[str, Any]:
        """Convert a batch of data to a tensor."""
        struct_batch_list = []
        input_list = []
        object_list = []
        true_node_cluster_ids_list = []
        partial_scene_cluster_ids_list = []
        surface_list = []

        # Unpack batch and convert to tensors.
        for key, scene_dict in batch.items():
            struct_batch_list.append((key, scene_dict))
            tensor_dict = self.scene_struct_to_tensor(
                scene_dict, shuffle=shuffle
            )
            input_list.append(tensor_dict["input"])
            object_list.append(tensor_dict["object_list"])

            true_node_cluster_ids_list.append(tensor_dict["true_node_cluster_ids"])
            partial_scene_cluster_ids_list.append(
                tensor_dict["partial_scene_cluster_ids"]
            )
            surface_list.append(tensor_dict["surface_list"])

        # Pad the list of input tensors.
        input_tensor, padding_mask, node_split = utils_torch.pad_tensor_from_list(
            input_list
        )

        return {
            "input": input_tensor,
            "padding_mask": padding_mask,
            "node_split": node_split,
            "true_node_cluster_ids": true_node_cluster_ids_list,  # True (goal) cluster IDs.
            "partial_scene_cluster_ids": partial_scene_cluster_ids_list,  # Partial cluster IDs
            "object_list": object_list,  # Object entities in the scene.
            "struct_batch": struct_batch_list,  # Scene structs and metadata (not dictionary!).
            "surface_list": surface_list,  # Ordering of surfaces.
        }

    def scene_struct_to_tensor(
        self, struct_dict: Dict[str, Any], shuffle: bool = True
    ) -> Dict[str, Any]:
        """Converts scene struct to a deconstructed scene graph."""
        partial_scene = struct_dict["partial"]
        goal_scene = struct_dict["goal"]
        surface_list = list(s.name for s in goal_scene)
        if shuffle:
            random.shuffle(surface_list)

        # Find cluster IDs for all the objects (including empty surfaces).
        placed_objectID_surf_pairs, unplaced_objectID_surf_pairs = (
            utils_data.find_all_object_placements(
                partial_scene, goal_scene, include_empty_surfaces=True
            )
        )
        if shuffle:
            random.shuffle(placed_objectID_surf_pairs)
            random.shuffle(unplaced_objectID_surf_pairs)
        # Pre-placed objects and their cluster IDs (1-indexed).
        object_list = list(t[0] for t in placed_objectID_surf_pairs)
        partial_cluster_ids = list(
            1 + surface_list.index(t[1]) for t in placed_objectID_surf_pairs
        )
        true_cluster_ids = list(
            1 + surface_list.index(t[1]) for t in placed_objectID_surf_pairs
        )
        # Unplaced objects and their cluster IDs.
        object_list.extend(list(t[0] for t in unplaced_objectID_surf_pairs))
        # TODO: issue with cluster ids not matching with original scenes.
        partial_cluster_ids.extend([0] * len(unplaced_objectID_surf_pairs))
        true_cluster_ids.extend(
            list(1 + surface_list.index(t[1]) for t in unplaced_objectID_surf_pairs)
        )
        assert len(object_list) == len(partial_cluster_ids) == len(true_cluster_ids)
        partial_cluster_ids = torch.tensor(partial_cluster_ids, device=DEVICE, dtype=torch.long)
        true_cluster_ids = torch.tensor(true_cluster_ids, device=DEVICE, dtype=torch.long)
        object_names = list(obj.name for obj in object_list)

        # Compute instance ids for objects.
        instance_ids = []
        object_instance_dict = {
            obj_name: list(range(object_names.count(obj_name)))
            for obj_name in set(object_names)
        }
        for i, obj_name in enumerate(object_names):
            inst_id = random.choice(object_instance_dict[obj_name])
            object_instance_dict[obj_name].remove(inst_id)
            instance_ids.append(inst_id)
        instance_ids = torch.tensor(instance_ids, device=DEVICE, dtype=torch.long)

        # Create input tensor.
        object_tensor = self.object_name_to_tensor(object_names)
        object_pos_instance = self.position_encoder_object(instance_ids.flatten())
        object_pos_surface = self.position_encoder_surface(partial_cluster_ids)
        input_tensor = torch.cat(
            [object_tensor, object_pos_instance, object_pos_surface], dim=-1
        )

        return {
            "input": input_tensor,
            "true_node_cluster_ids": true_cluster_ids,
            "partial_scene_cluster_ids": partial_cluster_ids,
            "surface_list": surface_list,
            "object_list": object_list,
        }
