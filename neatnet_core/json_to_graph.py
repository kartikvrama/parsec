"""Class to convert json data to tensor graph for training and evaluation."""

from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import random
import numpy as np
import torch

from utils import constants
from utils import data_struct
from utils import utils_data
from utils import utils_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_pos_vec(pos_list: List[float], mean: Tuple[float, float], norm: Tuple[float, float]):
    """Normalize position vector."""
    pos_vec = torch.tensor(pos_list, device=DEVICE, dtype=torch.double)
    assert pos_vec.shape[1] == 2 and len(pos_vec.shape) == 2, f"Invalid position vector shape: {pos_vec.shape}"
    mean_x, mean_y = mean
    norm_x, norm_y = norm
    if not norm_x:
        norm_x = 1
    if not norm_y:
        norm_y = 1
    return torch.vstack(
        [2*(pos_vec[:, 0] - mean_x) / norm_x, 2*(pos_vec[:, 1] - mean_y) / norm_y]
    ).T


class NeatNetBatchGen:
    """Class to generate tensor graph batches from JSON data."""

    def __init__(
        self,
        object_list: List[str],
        surface_constants: Dict[str, Dict[str, str]],
        object_embedding_dict: Optional[Dict[str, torch.Tensor]],
    ):
        """Initialize embedding generators for objects, surfaces, and container type."""
        self.object_list = object_list
        self.surface_constants = surface_constants

        # Initialize embedding generators.
        self.object_name_to_tensor = utils_torch.EmbeddingCaller(
            object_list, object_embedding_dict
        )
        self.object_dimension = self.object_name_to_tensor.get_dimension()

    def _objects_to_graph(
        self, object_list: List[data_struct.ObjectEntity]
    ) -> Dict[str, Any]:
        """Converts object list to a fully-connected graph.

        Positions are assigned dummy values since this is for evaluation.
        """
        obj_names = [obj.name for obj in object_list]
        obj_entities = object_list
        return {
            "objects": obj_names,
            "object_entities": obj_entities,
            "positions": [[9999, 9999]]*len(obj_names)
        }

    def _scene_struct_to_graph(
        self, scene_struct: List[data_struct.SurfaceEntity]
    ) -> Dict[str, Any]:
        """Converts scene structure to a fully-connected graph."""
        obj_names = []
        obj_entities = []
        positions = []
        for surface in scene_struct[:-1]:
            for obj in surface.objects_on_surface:
                obj_names.append(obj.name)
                obj_entities.append(obj)
                positions.append(surface.position)
        return {
            "objects": obj_names,
            "object_entities": obj_entities,
            "positions": positions,
        }

    def _create_edge_list(self, num_objects):
        """Creates a fully-connected edge list for a graph.
        
        Returns a tensor of shape (2, num_objects^2) where each column is an
        edge.
        """
        return torch.cartesian_prod(
            torch.arange(num_objects, device=DEVICE),
            torch.arange(num_objects, device=DEVICE),
        ).T

    def example_to_graph(
        self,
        struct_dict: Dict[str, Any],
        is_train: bool = True
    ):
        """Converts single example to graph."""
        observed_scene_list = deepcopy(struct_dict["observed_arrangement_list"])
        assert len(observed_scene_list) > 0
        goal_scene = deepcopy(struct_dict["goal"])

        # Convert scene structs into objects and positions.
        example_tuples = list(
            self._scene_struct_to_graph(scene) for scene in observed_scene_list
        )
        example_objects = [t["objects"] for t in example_tuples]
        example_obj_embeddings = list(
            torch.cat([self.object_name_to_tensor(n) for n in nodes], dim=0)
            for nodes in example_objects
        )
        example_positions = [t["positions"] for t in example_tuples]

        # Normalize graph positions.
        assert observed_scene_list[0][-1].surface_type == constants.SURFACE_TYPES.NOT_PLACED
        surface_positions = list(s.position for s in observed_scene_list[0][:-1])
        surface_position_dict = {
            s.name: torch.tensor(s.position, device=DEVICE, dtype=torch.double)
            for s in observed_scene_list[0][:-1]
        }
        x_positions, y_positions = zip(*surface_positions)
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        mean_x, mean_y = (x_positions.mean(), y_positions.mean())
        norm_x, norm_y = (x_positions.max(), y_positions.max())
        example_positions = list(
            _normalize_pos_vec(pos_list, (mean_x, mean_y), (norm_x, norm_y))
            for pos_list in example_positions
        )

        # Concatenate embeddings for example graphs.
        example_nodes_list = list(
            torch.cat([emb, pos], dim=-1)
            for emb, pos in zip(example_obj_embeddings, example_positions)
        )
        example_edges_list = list(
            self._create_edge_list(len(objs)) for objs in example_objects
        )
        (
            example_nodes_tensor,
            example_edges_tensor,
            example_scene_ids,
            example_nodes_by_scene,
        ) = utils_torch.merge_scene_graphs(example_nodes_list, example_edges_list)

        # Generate prediction graphs.
        (
            train_pred_nodes_tensor,
            train_pred_edges_tensor,
            train_pred_scene_ids,
            train_pred_nodes_by_scene,
        ) = utils_torch.merge_scene_graphs(example_obj_embeddings, example_edges_list)

        if is_train:
            eval_pred_dict = self._scene_struct_to_graph(goal_scene)
        else:
            unplaced_objects = deepcopy(struct_dict["partial"][-1].objects_on_surface)
            eval_pred_dict = self._objects_to_graph(unplaced_objects)
        
        eval_pred_obj_embeddings = torch.cat(
            [self.object_name_to_tensor(n) for n in eval_pred_dict["objects"]],
            dim=0
        )
        eval_pred_positions = _normalize_pos_vec(
            eval_pred_dict["positions"], (mean_x, mean_y), (norm_x, norm_y)
        )  # Normalize goal positions.
        eval_pred_edges = self._create_edge_list(len(eval_pred_dict["objects"]))

        return {
            "user_id": struct_dict["user_id"],
            "container": struct_dict["container_type"],
            "household": struct_dict["household"],
            "example_nodes": example_nodes_tensor,
            "example_edges": example_edges_tensor,
            "example_scene_ids": example_scene_ids,
            "example_nodes_by_scene": example_nodes_by_scene,
            "train_pred_nodes": train_pred_nodes_tensor,
            "train_pred_edges": train_pred_edges_tensor,
            "train_pred_scene_ids": train_pred_scene_ids,
            "train_pred_nodes_by_scene": train_pred_nodes_by_scene,
            "train_pred_positions": example_nodes_tensor[:, -2:].clone(),
            "eval_pred_object_entities": eval_pred_dict["object_entities"],
            "eval_pred_nodes": eval_pred_obj_embeddings,
            "eval_pred_edges": eval_pred_edges,
            "eval_pred_positions": eval_pred_positions,
            "eval_partial_scene": struct_dict["partial"],
            "eval_pred_scene": goal_scene,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "norm_x": norm_x,
            "norm_y": norm_y,
            "surface_position_dict": surface_position_dict,
        }

    def batch_to_tensor(self, batch, is_shuffle=True):
        """Converts batch of examples to tensors."""
        # TODO: Test for batch size > 2.
        if len(batch) > 2:
            raise NotImplementedError("Batching examples more then 2 is untested.")
        scene_struct_list = list(batch.values())
        graph_dict_list = list(
            self.example_to_graph(scene_struct, is_train=is_shuffle)
            for scene_struct in scene_struct_list
        )
        if len(batch) == 1:  # Return single graph for batch size 1.
            graph_dict = graph_dict_list[0]
            for prefix in ["example_", "train_pred_", "eval_pred_"]:
                graph_dict[f"{prefix}batch_ids"] = torch.zeros(
                    (len(graph_dict[f"{prefix}nodes"]),), device=DEVICE
                )
            return graph_dict
        if is_shuffle:
            random.shuffle(graph_dict_list)

        # Create lists of tensors for each graph.
        batch_dict = None
        for num, graph_dict in enumerate(graph_dict_list):
            if batch_dict is None:
                batch_dict = {
                    "container": graph_dict["container"],
                    "household": graph_dict["household"],
                    "mean_x": graph_dict["mean_x"],
                    "mean_y": graph_dict["mean_y"],
                    "norm_x": graph_dict["norm_x"],
                    "norm_y": graph_dict["norm_y"],
                }
            else:
                assert batch_dict["container"] == graph_dict["container"], "Container mismatch in batch"
                assert batch_dict["household"] == graph_dict["household"], "Household mismatch in batch"
                assert batch_dict["mean_x"] == graph_dict["mean_x"], "Mean x mismatch in batch"
                assert batch_dict["mean_y"] == graph_dict["mean_y"], "Mean y mismatch in batch"
                assert batch_dict["norm_x"] == graph_dict["norm_x"], "Norm x mismatch in batch"
                assert batch_dict["norm_y"] == graph_dict["norm_y"], "Norm y mismatch in batch"
            for key in [
                "user_id",
                "surface_position_dict",
                "eval_pred_object_entities",
                "eval_pred_scene",
                "eval_partial_scene",
            ]:
                if f"{key}_list" not in batch_dict:
                    batch_dict[f"{key}_list"] = []
                batch_dict[f"{key}_list"].append(graph_dict[key])
            for prefix in ["example_", "train_pred_", "eval_pred_"]:
                for suffix in ["nodes", "edges"]:  # Add nodes and edges.
                    if f"{prefix}{suffix}" not in batch_dict:
                        batch_dict[f"{prefix}{suffix}"] = []
                    batch_dict[f"{prefix}{suffix}"].append(graph_dict[f"{prefix}{suffix}"])
                if prefix != "example_":  # Add positions.
                    if f"{prefix}positions" not in batch_dict:
                        batch_dict[f"{prefix}positions"] = []
                    batch_dict[f"{prefix}positions"].append(graph_dict[f"{prefix}positions"])
                if prefix != "eval_pred_":  # Add scene ids and per scene nodes.
                    for suffix in ["scene_ids", "nodes_by_scene"]:
                        if f"{prefix}{suffix}" not in batch_dict:
                            batch_dict[f"{prefix}{suffix}"] = []
                        batch_dict[f"{prefix}{suffix}"].append(graph_dict[f"{prefix}{suffix}"])
                if f"{prefix}batch_ids" not in batch_dict:  # Add batch ids.
                    batch_dict[f"{prefix}batch_ids"] = torch.tensor(
                        [num]*len(graph_dict[f"{prefix}nodes"]),
                        device=DEVICE,
                        dtype=torch.long
                    )
                else:
                    batch_dict[f"{prefix}batch_ids"] = torch.cat(
                        [
                            batch_dict[f"{prefix}batch_ids"],
                            torch.tensor(
                                [num]*len(graph_dict[f"{prefix}nodes"]),
                                device=DEVICE,
                                dtype=torch.long
                            )
                        ],
                        dim=0
                    )

        # Concatenate tensors for each graph.
        for prefix in ["example_", "train_pred_", "eval_pred_"]:
            for suffix in ["scene_ids", "nodes_by_scene", "positions"]:
                if f"{prefix}{suffix}" not in batch_dict:
                    continue
                batch_dict[f"{prefix}{suffix}"] = torch.cat(batch_dict[f"{prefix}{suffix}"], dim=0)
            # Merge edge matrices diagonally.
            combined_nodes, combined_edges, _, nodes_by_user = utils_torch.merge_scene_graphs(
                batch_dict[f"{prefix}nodes"], batch_dict[f"{prefix}edges"]
            )
            batch_dict[f"{prefix}nodes"] = combined_nodes
            batch_dict[f"{prefix}edges"] = combined_edges
            batch_dict[f"{prefix}nodes_by_user"] = nodes_by_user
            if prefix != "eval_pred_":
                assert (
                    len(batch_dict[f"{prefix}nodes"])
                    == len(batch_dict[f"{prefix}batch_ids"])
                    == len(batch_dict[f"{prefix}scene_ids"])
                ), f"Number of nodes mismatch in {prefix} tensors."
        return batch_dict
