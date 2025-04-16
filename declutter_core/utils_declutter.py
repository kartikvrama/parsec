"""Helper functions for declutter model."""
from typing import List, Dict, Any, Callable, Union, Tuple
from copy import deepcopy
import random
import torch
from torch.nn.utils.rnn import pad_sequence

from utils import constants
from utils import data_struct
from utils import utils_data
from utils import utils_torch

# Constants.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: convert to data class.
class ObjectSurfaceTensor():
    """Class to store object and surface tensors."""
    object_class: torch.Tensor
    object_copy_index: List[int]
    surface_type: torch.Tensor
    grid: torch.Tensor
    real: List[int]

    def __init__(
        self, object_class: torch.Tensor, object_copy_index: List[int],
        surface_type: torch.Tensor, grid: torch.Tensor, real: List[int]
    ):
        """Initializes the ObjectSurfaceTensor object and validates inputs."""
        assert all(
            len(x.size()) == len(object_class.size()) for x in [object_class, surface_type, grid]
        )
        if not object_class.size()[:-1] == surface_type.size()[:-1] == grid.size()[:-1]:
            raise ValueError("All tensors must have the same first (n-1) dimension.")
        if not len(real) == len(object_copy_index) == len(object_class):
            raise ValueError(f"Input lengths do not match: {len(real)}, {len(object_copy_index)}, {object_class.size()[0]}.")
        if len(object_class.size()) > 2:
            if not all(
                len(real[i]) == len(object_copy_index[i]) == len(object_class[i])
                for i in range(object_class.size()[0])
            ):
                raise ValueError(
                    f"Input lengths do not match: {[len(r) for r in real]}, {[len(oc) for oc in object_copy_index]}, {object_class.size()}."
                )
        self.object_class = object_class
        self.surface_type = surface_type.to(torch.long)
        self.grid = grid
        if len(object_class.size()) == 1:
            self.object_class = self.object_class.unsqueeze(0)
            self.surface_type = self.surface_type.unsqueeze(0)
            self.grid = self.grid.unsqueeze(0)
        self.object_copy_index = object_copy_index
        self.real = real

    def __len__(self):
        return self.object_class.size()[0]

    def __copy__(self):
        return ObjectSurfaceTensor(
            object_class=self.object_class.clone(),
            object_copy_index=list(self.object_copy_index),
            surface_type=self.surface_type.clone(),
            grid=self.grid.clone(),
            real=list(self.real)
        )

    def get_dimension(self):
        """Returns the final dimension of the full tensor."""
        return (
            self.object_class.size()[-1] + self.surface_type.size()[-1]
            + self.grid.size()[-1]
        )


def stack_object_surface_tensors(
    tensor_list: List[ObjectSurfaceTensor]
) -> ObjectSurfaceTensor:
    """Stacks a list of ObjectSurfaceTensor objects."""
    return ObjectSurfaceTensor(
        object_class=torch.stack([x.object_class for x in tensor_list]),
        object_copy_index=[x.object_copy_index for x in tensor_list],
        surface_type=torch.stack([x.surface_type for x in tensor_list]),
        grid=torch.stack([x.grid for x in tensor_list]),
        real=[x.real for x in tensor_list]
    )


def concatenate_object_surface_tensors(
    tensor_list: List[ObjectSurfaceTensor]
) -> ObjectSurfaceTensor:
    """Concatenates a list of ObjectSurfaceTensor objects along the first dimension."""
    copy_index_concatenated = []
    for x in tensor_list:
        copy_index_concatenated.extend(x.object_copy_index)
    real_concatenated = []
    for x in tensor_list:
        real_concatenated.extend(x.real)
    return ObjectSurfaceTensor(
        object_class=torch.cat([x.object_class for x in tensor_list], dim=0),
        object_copy_index=copy_index_concatenated,
        surface_type=torch.cat([x.surface_type for x in tensor_list], dim=0),
        grid=torch.cat([x.grid for x in tensor_list], dim=0),
        real=real_concatenated
    )


def pad_object_surface_tensors(tensor_list: List[ObjectSurfaceTensor]) -> ObjectSurfaceTensor:
    """Pads a list of 2D ObjectSurfaceTensor objects."""
    object_class, padding_mask, _ = utils_torch.pad_tensor_from_list(
        [x.object_class for x in tensor_list]
    )
    surface_type = pad_sequence(
        [x.surface_type for x in tensor_list], batch_first=True
    ).to(torch.double)
    grid = pad_sequence(
        [x.grid for x in tensor_list], batch_first=True
    ).to(torch.double)
    return (
        ObjectSurfaceTensor(
            object_class=object_class,
            object_copy_index = [
                x.object_copy_index + [0]*(padding_mask.size()[1] - len(x.object_copy_index))
                for x in tensor_list
            ],
            surface_type=surface_type, grid=grid,
            real=[
                x.real + [0]*(padding_mask.size()[1] - len(x.real)) for x in tensor_list
            ]
        ),
        padding_mask
    )


def split_object_surface_latents(
    object_surface_latents: torch.Tensor,
    surface_split_mask: torch.Tensor,
    padding_mask: torch.Tensor
):
    """Splits combined output tensor into object and surface latents."""
    assert len(object_surface_latents.size()) == 2
    assert object_surface_latents.size()[0] == surface_split_mask.size()[0]
    assert surface_split_mask.size() == padding_mask.size()
    object_split_mask = (1 - surface_split_mask) * (1 - padding_mask)
    surface_latent = object_surface_latents[torch.nonzero(surface_split_mask).squeeze(), :]
    if len(surface_latent.size()) == 1:
        surface_latent = surface_latent.unsqueeze(0)
    object_latent = object_surface_latents[torch.nonzero(object_split_mask).squeeze(), :]
    if len(object_latent.size()) == 1:
        object_latent = object_latent.unsqueeze(0)
    return object_latent, surface_latent


def scene_to_array(
        scene: List[data_struct.SurfaceEntity], skip_empty: bool):
    """Deconstructs an arranged scene into lists of features.
    
    Note: This function ignores objects on the unplaced (i.e., last) surface.
    """
    object_labels = []
    placement_type_labels = []
    placement_grid_labels = []
    placement_name_labels = []
    for surface in scene[:-1]:
        if len(surface.objects_on_surface) > 0:
            object_labels.extend(
                [obj.name for obj in surface.objects_on_surface]
            )
            placement_type_labels.extend(
                [surface.surface_type for _ in surface.objects_on_surface]
            )
            placement_grid_labels.extend(
                [surface.position for _ in surface.objects_on_surface]
            )
            placement_name_labels.extend(
                [surface.name for _ in surface.objects_on_surface]
            )
        elif not skip_empty:
            object_labels.append(constants.EMPTY_LABEL)
            placement_type_labels.append(surface.surface_type)
            placement_grid_labels.append(surface.position)
            placement_name_labels.append(surface.name)

    count_labels_per_object = {
        ob: list(range(object_labels.count(ob))) for ob in set(object_labels)
    }
    object_copy_indices = []
    for ob in object_labels:
        cid = random.choice(count_labels_per_object[ob])
        object_copy_indices.append(cid)
        count_labels_per_object[ob].remove(cid)

    return (
        object_labels, object_copy_indices, placement_type_labels,
        placement_grid_labels, placement_name_labels
    )


def grid_to_tensor(grid_list, index_to_embb):
    """Converts xy coordinates into positional embeddings."""
    if all(isinstance(x, int) for x in grid_list):
        grid_list = [grid_list]
    assert all(len(x) == 2 for x in grid_list), "Grid list must be a list of 2-tuples."
    embb_x = index_to_embb(torch.tensor([t[0] for t in grid_list]))
    embb_y = index_to_embb(torch.tensor([t[1] for t in grid_list]))
    embb = torch.zeros((embb_x.size()[0], 2*embb_x.size()[1]), device=DEVICE)
    embb[:, ::2] = embb_x
    embb[:, 1::2] = embb_y
    return embb


def scene_to_tensor(
    scene: List[data_struct.SurfaceEntity],
    surface_list: List[str],
    object_to_embedding: Callable[[List[str]], torch.Tensor],
    placement_type_to_embedding: Callable[[List[str]], torch.Tensor],
    placement_grid_to_embedding: Callable[[Union[int, List[int]]], torch.Tensor],
    skip_empty: bool, shuffle: bool, device=DEVICE
) -> Tuple[ObjectSurfaceTensor, torch.Tensor, torch.Tensor, List[str]]:
    """Converts a demonstration scene to a tensor.
    
    This function wraps around scene_to_array and generates cluster labels and
    masks. Unplaced objects are ignored.
    """

    (
        object_labels, object_copy_indices, placement_type_labels, placement_grid_labels,
        placement_name_labels
    ) = scene_to_array(scene, skip_empty=skip_empty)
    object_tensor = object_to_embedding(object_labels)
    placement_type_tensor = placement_type_to_embedding(placement_type_labels)
    placement_grid_tensor = grid_to_tensor(
        placement_grid_labels, placement_grid_to_embedding
    )
    if shuffle:
        indices = torch.randperm(object_tensor.size(0))
        object_labels = list(object_labels[i] for i in indices)
        object_copy_indices = list(object_copy_indices[i] for i in indices)
        object_tensor = object_tensor[indices]
        placement_type_tensor = placement_type_tensor[indices]
        placement_grid_tensor = placement_grid_tensor[indices]
        placement_name_labels = list(placement_name_labels[i] for i in indices)
    obj_surf_ts = ObjectSurfaceTensor(
        object_class=object_tensor, object_copy_index=object_copy_indices,
        surface_type=placement_type_tensor, grid=placement_grid_tensor,
        real=[1]*len(object_labels)
    )
    cluster_labels = torch.tensor([
        surface_list.index(surface_name) for surface_name in placement_name_labels
    ]).to(device)
    cluster_mask = torch.cat([
        torch.where(cluster_labels == cluster_id, 1, 0).unsqueeze(1)
        for cluster_id in torch.arange(len(surface_list))
    ], dim=1).to(device=device, dtype=torch.double) # (num_objects x num_clusters)
    return obj_surf_ts, cluster_labels, cluster_mask, object_labels


def scene_list_to_tensor(
    scene_list: List[List[data_struct.SurfaceEntity]],
    surface_list: List[str],
    object_to_embedding: Callable[[List[str]], torch.Tensor],
    placement_type_to_embedding: Callable[[List[str]], torch.Tensor],
    placement_grid_to_embedding: Callable[[Union[int, List[int]]], torch.Tensor],
    shuffle: bool, skip_empty:bool, device=DEVICE
) -> Tuple[List[ObjectSurfaceTensor], List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    """Converts a list of scenes to tensors."""
    tensor_list = []
    cluster_label_list = []
    cluster_mask_list = []
    object_label_list = []
    if shuffle:
        random.shuffle(scene_list)
    for scene in scene_list:
        tensor, cluster_labels, cluster_mask, object_labels = scene_to_tensor(
            scene, surface_list, object_to_embedding, placement_type_to_embedding,
            placement_grid_to_embedding, shuffle=shuffle, skip_empty=skip_empty,
            device=device
        )
        tensor_list.append(tensor)
        cluster_label_list.append(cluster_labels)
        cluster_mask_list.append(cluster_mask)
        object_label_list.append(object_labels)
    return tensor_list, cluster_label_list, cluster_mask_list, object_label_list
