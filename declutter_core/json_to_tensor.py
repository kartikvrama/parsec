"""Class to convert json data to tensor for training and evaluation."""
from typing import Any, Dict, List, Optional
from copy import copy, deepcopy
import random
import numpy as np

import torch
from torch.nn import functional as F

from utils import constants
from utils import utils_data
from utils import utils_torch
from declutter_core import utils_declutter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeclutterBatchGen():
    """Class to generate tensor batches from JSON data."""
    def __init__(
        self,
        object_list: List[str],
        surface_constants: Dict[str, Dict[str, str]],
        object_embedding_dict: Optional[Dict[str, torch.Tensor]],
        grid_dimension: int
    ):
        """Initialize embedding generators for objects, surfaces, and container type.
        """
        self.object_list = object_list
        self.surface_constants = surface_constants
        self.grid_dimension = grid_dimension

        # Initialize embedding generators.
        self.object_name_to_tensor = utils_torch.EmbeddingCaller(
            object_list, object_embedding_dict
        )
        self.object_dimension = self.object_name_to_tensor.get_dimension()
        self.surface_type_to_index = utils_torch.EmbeddingCaller(
            label_list=constants.SURFACE_TYPES.SURFACE_TYPE_LIST  + [constants.SURFACE_TYPES.NOT_PLACED]
        )
        self.container_type_to_index = utils_torch.EmbeddingCaller(
            label_list=constants.ENVIRONMENTS.ENVIRONMENT_LIST
        )

        self.placement_grid_to_sinusoidal  = utils_torch.PositionalEncodingCaller(
            max_position=16, d_model=grid_dimension//2
        )

    def _pad_mask(self, mask_list: List[torch.Tensor]) -> torch.Tensor:
        """Pads a list of masks with zeros and stacks them."""
        max_rows = max(m.size()[0] for m in mask_list)
        max_cols = max(m.size()[1] for m in mask_list)
        result = torch.stack([
            F.pad(
                input=mask,
                pad=(0, max_cols - mask.size()[1], 0, max_rows - mask.size()[0]),
                value=0
            )
            for mask in mask_list
        ], dim=0).to(mask_list[0].dtype).to(DEVICE)
        if len(mask_list) == 1 and len(result.size()) == 2:
            result = result.unsqueeze(0)
        return result

    def batch_to_tensor(self, batch: Dict[str, Any], is_shuffle: bool = True) -> Dict[str, Any]:
        """Generate tensors from a batch of scene structs."""
        scene_ids = []
        batch_list = {}
        partial_scene_list = []
        goal_scene_list = []
        observed_scene_list = []
        for sid, struct_dict in batch.items():
            partial_scene_list.append(struct_dict["partial"])
            goal_scene_list.append(struct_dict["goal"])
            observed_scene_list.append(struct_dict["observed_arrangement_list"])
            scene_ids.append(sid)
            tensor_dict = self.scene_struct_to_tensor(struct_dict, shuffle=is_shuffle)
            if tensor_dict is None:
                continue
            for key, value in tensor_dict.items():
                if key not in batch_list:
                    batch_list[key] = [value]
                else:
                    batch_list[key].append(value)
        seq_tensor_padded, seq_padding_mask = utils_declutter.pad_object_surface_tensors(
            batch_list["seq_tensor"]
        )
        combined_scene_tensor_padded, combined_scene_padding_mask = utils_declutter.pad_object_surface_tensors(
            batch_list["combined_scene_tensor"]
        )
        return {
            "scene_ids": scene_ids,
            "seq_ostensor": seq_tensor_padded,
            "seq_demo_index": self._pad_mask(batch_list["seq_demo_index"]).squeeze(-1),
            "seq_padding_mask": seq_padding_mask,
            "container_type_tensor": torch.concat(batch_list["container_type_tensor"], dim=0).to(DEVICE),
            "seq_cluster_labels": batch_list["seq_cluster_labels"],
            "seq_object_list": batch_list["seq_object_list"],
            "seq_cluster_mask": self._pad_mask(batch_list["seq_cluster_mask"]),
            "seq_split_mask": self._pad_mask(batch_list["seq_split_mask"]).squeeze(-1),
            "seq_split_mask_unpadded": list(t.squeeze() for t in batch_list["seq_split_mask"]),
            "scene_ostensor": combined_scene_tensor_padded,
            "scene_padding_mask": combined_scene_padding_mask,
            "partial_scene_split_mask": self._pad_mask(batch_list["pscene_split_mask"]).squeeze(-1),
            "partial_scene_cluster_mask": self._pad_mask(batch_list["pscene_cluster_mask"]),
            "object_triplet_labels": batch_list["object_triplet_labels"],
            "partial_sequence_list": batch_list["partial_sequence_list"],
            "partial_scene_list": partial_scene_list,
            "goal_scene_list": goal_scene_list,
            "observed_scene_list": observed_scene_list,
            "unplaced_object_ids": batch_list["unplaced_object_ids"]
        }

    def scene_struct_to_tensor(self, struct_dict: Dict[str, Any], shuffle: bool = True) -> torch.Tensor:
        """Converts scene struct to an object-centric tensor.
        """
        observed_scene_list = deepcopy(struct_dict["observed_arrangement_list"])
        partial_scene = deepcopy(struct_dict["partial"])
        goal_scene = deepcopy(struct_dict["goal"])
        container_type_tensor = self.container_type_to_index(
            struct_dict["container_type"]
        ).to(torch.long)
        if (
            partial_scene[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED
            or goal_scene[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED
            or any(
                ds[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED
                for ds in observed_scene_list
            )
        ):
            raise ValueError("Last surface is not NOT_PLACED.")
        if any(s.objects_on_surface for s in partial_scene[:-1]):
            # If partial scene is not empty.
            observed_scene_list.insert(0, partial_scene)

        # List of surfaces for consistency.
        surface_list = list(s.name for s in goal_scene)

        # Convert observed scenes to a sequence tensor.
        (
            seq_tensor_list, seq_cluster_label_list, seq_cluster_mask_list,
            seq_object_list
        ) = self._observed_scenes_to_tensor(
            observed_scene_list, surface_list, is_shuffle=shuffle
        )
        if not seq_cluster_label_list:
            raise ValueError("Found example with no observed scenes.")
        elif len(seq_cluster_label_list) == 1:
            seq_cluster_labels = seq_cluster_label_list[0]
        else:
            seq_cluster_labels = torch.cat(seq_cluster_label_list, dim=0)
        seq_tensor, seq_cluster_mask, seq_split_mask, seq_demo_index = self._seq_list_to_tensor(
            seq_tensor_list, seq_cluster_mask_list
        )

        # Convert the partially arranged scene to a sequence tensor.
        (
            pscene_tensor, pscene_cluster_labels, pscene_cluster_mask,
            pscene_object_list
        ) = self._partial_scene_to_tensor(partial_scene, surface_list, is_shuffle=shuffle)

        # Fetch unplaced object ids and tensors.
        unplaced_object_ids = [
            obj.object_id for obj in partial_scene[-1].objects_on_surface
        ]
        unplaced_object_labels = [
            obj.name for obj in partial_scene[-1].objects_on_surface
        ]
        if shuffle:
            shuffle_indices = np.random.permutation(len(unplaced_object_ids))
            unplaced_object_ids = [unplaced_object_ids[idx] for idx in shuffle_indices]
            unplaced_object_labels = [unplaced_object_labels[idx] for idx in shuffle_indices]

        # Match object ids with their true placements.
        unplaced_objID_surf_pairs = utils_data.find_unplaced_object_placement(
            partial_scene, goal_scene
        )
        unplaced_objID_surfID_pairs = list(
            (obj_id, surface_list.index(surf_id))
            for obj_id, surf_id in unplaced_objID_surf_pairs
        )
        temp_assignments_list = list(unplaced_objID_surfID_pairs)
        unplaced_obj_cluster_indices = []
        for oid in unplaced_object_ids:
            matching_assignment = [
                asgn for asgn in temp_assignments_list if asgn[0] == oid
            ]
            if not matching_assignment:
                raise ValueError(
                    f"Could not find true placement for unplaced object {oid},"
                    f" please check scene {struct_dict['scene_id']}."
                )
            matching_assignment = matching_assignment[0]
            temp_assignments_list.remove(matching_assignment)
            unplaced_obj_cluster_indices.append(matching_assignment[1])
        assert not temp_assignments_list
        unplaced_obj_cluster_indices = torch.tensor(
            unplaced_obj_cluster_indices, device=DEVICE
        )

        # [CLUSTER] Concatenate the partial scene tensor and unplaced object
        # tensors.
        unplaced_obj_tensor_stacked = self._unplaced_objects_to_tensor(
            unplaced_object_labels, pscene_object_list
        )
        combined_scene_tensor = utils_declutter.concatenate_object_surface_tensors(
            [pscene_tensor, unplaced_obj_tensor_stacked]
        )
        pscene_split_mask = torch.cat([
            torch.ones((len(pscene_tensor), 1), dtype=torch.double),
            torch.zeros((len(unplaced_obj_tensor_stacked), 1), dtype=torch.double)
        ], dim=0).to(DEVICE)

        object_triplet_labels = torch.cat([
            pscene_cluster_labels, unplaced_obj_cluster_indices
        ])
        return {
            "seq_tensor": seq_tensor,
            "seq_demo_index": seq_demo_index,
            "container_type_tensor": container_type_tensor,
            "seq_split_mask": seq_split_mask,
            "seq_cluster_labels": seq_cluster_labels,
            "seq_object_list": seq_object_list,
            "seq_cluster_mask": seq_cluster_mask,
            "combined_scene_tensor": combined_scene_tensor,
            "pscene_split_mask": pscene_split_mask,
            "pscene_cluster_mask": pscene_cluster_mask,
            "object_triplet_labels": object_triplet_labels,
            "partial_sequence_list": pscene_object_list + unplaced_object_labels,
            "unplaced_object_ids": unplaced_object_ids
        }

    def _add_seq_token(self, obj_surf_ts: utils_declutter.ObjectSurfaceTensor):
        """Adds an end token (zero tensor) to a 2D sequence tensor"""
        zero_tensor = utils_declutter.ObjectSurfaceTensor(
            object_class=torch.zeros_like(obj_surf_ts.object_class)[0].unsqueeze(0),
            object_copy_index=[0],
            surface_type=torch.zeros_like(obj_surf_ts.surface_type)[0].unsqueeze(0),
            grid=torch.zeros_like(obj_surf_ts.grid)[0].unsqueeze(0), real=[0]
        )
        return utils_declutter.concatenate_object_surface_tensors(
            [obj_surf_ts, zero_tensor]
        )

    def _seq_list_to_tensor(self, obj_surf_ts_list, cluster_mask_list):
        split_mask = None
        final_tensor = None
        final_cluster_mask = torch.empty((0, cluster_mask_list[0].size()[1]), device=DEVICE)
        # Add zeros tensor as sequence end token and concatenate tensors.
        demo_index_list = torch.empty((0,), device=DEVICE)
        for i, (ostensor, cmask) in enumerate(zip(obj_surf_ts_list, cluster_mask_list)):
            if final_tensor is None:
                final_tensor = copy(ostensor)
            else:
                final_tensor = utils_declutter.concatenate_object_surface_tensors(
                    [final_tensor, ostensor]
                )
            demo_index_list = torch.cat([
                demo_index_list, torch.tensor([i]*len(ostensor), device=DEVICE)
            ])
            if i < len(obj_surf_ts_list) - 1:
                final_tensor = self._add_seq_token(final_tensor)
                demo_index_list = torch.cat([
                    demo_index_list, torch.zeros((1,), device=DEVICE)
                ])
            final_cluster_mask = torch.vstack([
                final_cluster_mask, cmask, torch.zeros_like(cmask)[0].unsqueeze(0)
            ])
            split_mask_i = torch.tensor([1]*len(ostensor) + [0], device=DEVICE)
            if split_mask is None:
                split_mask = copy(split_mask_i)
            else:
                split_mask = torch.cat([split_mask, split_mask_i])
        return (
            final_tensor, final_cluster_mask, split_mask[:-1].unsqueeze(1),
            demo_index_list.unsqueeze(1).to(dtype=torch.long)
        )

    def _observed_scenes_to_tensor(self, observed_scene_list, surface_list, is_shuffle):
        return utils_declutter.scene_list_to_tensor(
            scene_list=observed_scene_list,
            surface_list=surface_list,
            object_to_embedding=self.object_name_to_tensor,
            placement_type_to_embedding=self.surface_type_to_index,
            placement_grid_to_embedding=self.placement_grid_to_sinusoidal,
            skip_empty=False, shuffle=is_shuffle, device=DEVICE
        )

    def _partial_scene_to_tensor(self, partial_scene, surface_list, is_shuffle):
        return utils_declutter.scene_to_tensor(
            scene=partial_scene,
            surface_list=surface_list,
            object_to_embedding=self.object_name_to_tensor,
            placement_type_to_embedding=self.surface_type_to_index,
            placement_grid_to_embedding=self.placement_grid_to_sinusoidal,
            skip_empty=False, shuffle=is_shuffle, device=DEVICE
        )

    def _unplaced_objects_to_tensor(
        self, unplaced_object_labels: List[str], placed_object_labels: List[str]
    ):
        object_tensor = self.object_name_to_tensor(unplaced_object_labels).to(DEVICE)
        surface_tensor = self.surface_type_to_index(
            [constants.SURFACE_TYPES.NOT_PLACED]*len(unplaced_object_labels)
        ).to(DEVICE)
        count_labels_dict = {
            ob: np.arange(
                    placed_object_labels.count(ob),
                    placed_object_labels.count(ob) + unplaced_object_labels.count(ob)
                )
            for ob in unplaced_object_labels
        }
        object_copy_indices = []
        for ob in unplaced_object_labels:
            cid = random.choice(count_labels_dict[ob])
            object_copy_indices.append(cid)
            idx_remove = np.where(count_labels_dict[ob] == cid)[0]
            np.delete(count_labels_dict[ob], idx_remove)
        grid_tensor = torch.zeros(
            (len(unplaced_object_labels), self.grid_dimension), device=DEVICE
        )
        return utils_declutter.ObjectSurfaceTensor(
            object_tensor, object_copy_indices, surface_tensor, grid_tensor,
            real=[1]*len(unplaced_object_labels)
        )

    def get_feature_length(self):
        """Returns the length of the node feature vector."""
        return {
            "object_dimension": self.object_dimension,
            "container_dimension": len(constants.ENVIRONMENTS.ENVIRONMENT_LIST),
            "surface_type_dimension": len(constants.SURFACE_TYPES),
            "surface_grid_dimension": self.grid_dimension,
            "type_dimension": 1
        }
