"""Functions to compare scene structs and project using MDS."""
from typing import Any, List, Dict, Mapping, Tuple
from copy import deepcopy
import numpy as np
from sklearn.manifold import MDS
import scipy.optimize as spo

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from utils import data_struct
from utils import utils_data

def return_average_similarity_objects(wn_list_a: List[Any], wn_list_b: List[Any]):
    sim = 0
    if not wn_list_a or not wn_list_b:
        if not wn_list_a and not wn_list_b:
            return None
        return 0
    for wn_a in wn_list_a:
        for wn_b in wn_list_b:
            sim += wn_a.wup_similarity(wn_b)
    return sim / (len(wn_list_a) * len(wn_list_b))


def return_sim_scenes(
    scene_a: List[data_struct.SurfaceEntity],
    scene_b: List[data_struct.SurfaceEntity],
    wordnet_labels: Dict[str, str],
    objectId_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    # Note: last surface is not unplaced surface.
    surface_names_a = list(s.name for s in scene_a)
    surface_names_b = list(s.name for s in scene_b)
    sim = 0
    num_non_empty_surfaces = 0
    for i, sname in enumerate(surface_names_a):
        wn_list_a = list(
            wn.synset(wordnet_labels[objectId_dict[o.object_id]["name"]])
            for o in scene_a[i].objects_on_surface
        )
        wn_list_b = list(
            wn.synset(wordnet_labels[objectId_dict[o.object_id]["name"]])
            for o in scene_b[surface_names_b.index(sname)].objects_on_surface
        )
        if ignore_duplicates:
            measure = return_average_similarity_objects(set(wn_list_a), set(wn_list_b))
        else:
            measure = return_average_similarity_objects(wn_list_a, wn_list_b)
        if measure is None:
            continue
        sim += measure
        num_non_empty_surfaces += 1
    return sim/num_non_empty_surfaces


def return_sim_matrix(
    arrangements: List[List[data_struct.SurfaceEntity]],
    label_dict: Dict[str, str],
    object_id_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    sim_matrix = np.zeros((len(arrangements), len(arrangements)))
    for i, scene_a in enumerate(arrangements):
        for j, scene_b in enumerate(arrangements):
            if i == j:
                sim_matrix[i, j] = 1
            else:
                sim_matrix[i, j] = return_sim_scenes(
                    scene_a,
                    scene_b,
                    label_dict,
                    object_id_dict,
                    ignore_duplicates=ignore_duplicates,
                )
    return sim_matrix


def return_mds_projection(
    arrangements: List[List[data_struct.SurfaceEntity]],
    label_dict: Dict[str, str],
    dist_func: Mapping[
        Tuple[
            List[data_struct.SurfaceEntity],
            List[data_struct.SurfaceEntity],
            Dict[str, str]
        ], float
    ],
    objectId_dict: Dict[str, Dict[str, Any]],
    metric_mds:bool =True,
    eps: float = 1e-3,
    max_iter: int = 1000,
    ignore_duplicates: bool = False,
    **kwargs
):
    dist_matrix = np.zeros((len(arrangements), len(arrangements)))
    for i, scene_a in enumerate(arrangements):
        for j, scene_b in enumerate(arrangements):
            if i == j:
                continue
            else:
                # utils_data.visualize_scene(scene_a)
                # utils_data.visualize_scene(scene_b)
                dist_matrix[i, j] = dist_func(
                    scene_a,
                    scene_b,
                    label_dict,
                    objectId_dict=objectId_dict,
                    ignore_duplicates=ignore_duplicates,
                    **kwargs
                )
                # print(f"Distance between {i} and {j}: {dist_matrix[i, j]}")
    if metric_mds:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=True,
            verbose=1,
            eps=eps,
            max_iter=max_iter,
        )
    else:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=False,
            normalized_stress=True,
            verbose=1,
            eps=eps,
            max_iter=max_iter,
        )
    return dist_matrix, mds.fit_transform(dist_matrix)


def return_dist_scenes(
    scene_a: List[data_struct.SurfaceEntity],
    scene_b: List[data_struct.SurfaceEntity],
    wordnet_labels: Dict[str, str],
    objectId_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    average_sim = return_sim_scenes(
        scene_a,
        scene_b,
        wordnet_labels,
        objectId_dict,
        ignore_duplicates=ignore_duplicates,
    )
    return 1 - average_sim


def return_dist_scenes_clustered(
    scene_a: List[data_struct.SurfaceEntity],
    scene_b: List[data_struct.SurfaceEntity],
    category_label_dict: Dict[str, str],
    objectId_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    # Note: last surface is not unplaced surface.
    surface_names_a = list(s.name for s in scene_a)
    surface_names_b = list(s.name for s in scene_b)
    dist = 0
    num_non_empty_surfaces = 0
    for i, sname in enumerate(surface_names_a):
        cats_a = list(
            category_label_dict[objectId_dict[o.object_id]["name"]]
            for o in scene_a[i].objects_on_surface
        )
        cats_b = list(
            category_label_dict[objectId_dict[o.object_id]["name"]]
            for o in scene_b[surface_names_b.index(sname)].objects_on_surface
        )
        if ignore_duplicates:
            wn_list_a = list(wn.synset(c) for c in set(cats_a))
            wn_list_b = list(wn.synset(c) for c in set(cats_b))
        else:
            wn_list_a = list(wn.synset(c) for c in cats_a)
            wn_list_b = list(wn.synset(c) for c in cats_b)
        sim = return_average_similarity_objects(wn_list_a, wn_list_b)
        if sim is None:
            continue
        dist += 1 - sim
        num_non_empty_surfaces += 1
    return dist/num_non_empty_surfaces


def return_dist_scenes_leastmatching(
    scene_a: List[data_struct.SurfaceEntity],
    scene_b: List[data_struct.SurfaceEntity],
    wordnet_labels: Dict[str, str],
    objectId_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    # Note: last surface is not unplaced surface.
    surface_names_a = list(s.name for s in scene_a)
    surface_names_b = list(s.name for s in scene_b)
    sim_matrix = np.zeros((len(scene_a), len(scene_b)))
    for i, _ in enumerate(surface_names_a):
        wn_list_a = list(
            wn.synset(wordnet_labels[objectId_dict[o.object_id]["name"]])
            for o in scene_a[i].objects_on_surface
        )
        for j, _ in enumerate(surface_names_b):
            wn_list_b = list(
                wn.synset(wordnet_labels[objectId_dict[o.object_id]["name"]])
                for o in scene_b[j].objects_on_surface
            )
            if ignore_duplicates:
                sim = return_average_similarity_objects(set(wn_list_a), set(wn_list_b))
            else:
                sim = return_average_similarity_objects(wn_list_a, wn_list_b)
            if sim is not None:
                sim_matrix[i, j] = sim

    row_ind, col_ind = spo.linear_sum_assignment(sim_matrix, maximize=True)
    dist = 0
    num_non_empty_surfaces = 0
    for surf_id_a, surf_id_b in zip(row_ind, col_ind):
        if not scene_a[surf_id_a].objects_on_surface and not scene_b[surf_id_b].objects_on_surface:
            continue
        dist += 1 - sim_matrix[surf_id_a, surf_id_b]
        num_non_empty_surfaces += 1            
    return dist/num_non_empty_surfaces


def return_dist_scenes_leastmatching_wclusters(
    scene_a: List[data_struct.SurfaceEntity],
    scene_b: List[data_struct.SurfaceEntity],
    category_label_dict: Dict[str, str],
    objectId_dict: Dict[str, Dict[str, Any]],
    ignore_duplicates: bool = False,
):
    # Note: last surface is not unplaced surface.
    surface_names_a = list(s.name for s in scene_a)
    surface_names_b = list(s.name for s in scene_b)
    sim_matrix = np.zeros((len(scene_a), len(scene_b)))
    list_cats_a = list(None for _ in range(len(scene_a)))
    list_cats_b = list(None for _ in range(len(scene_b)))
    for i, _ in enumerate(surface_names_a):
        wn_cats_a = list(
            wn.synset(category_label_dict[objectId_dict[o.object_id]["name"]])
            for o in scene_a[i].objects_on_surface
        )
        # print(list_cats_a, list_cats_b)
        # print(wn_cats_a)
        if list_cats_a[i] is None:
            list_cats_a[i] = [c.name() for c in wn_cats_a]
        for j, _ in enumerate(surface_names_b):
            wn_cats_b = list(
                wn.synset(category_label_dict[objectId_dict[o.object_id]["name"]])
                for o in scene_b[j].objects_on_surface
            )
            # print(list_cats_a, list_cats_b)
            # print(wn_cats_b)
            if list_cats_b[j] is None:
                list_cats_b[j] = [c.name() for c in wn_cats_b]
            if ignore_duplicates:
                sim = return_average_similarity_objects(set(wn_cats_a), set(wn_cats_b))
            else:
                sim = return_average_similarity_objects(wn_cats_a, wn_cats_b)
            if sim is not None:
                sim_matrix[i, j] = sim

    row_ind, col_ind = spo.linear_sum_assignment(sim_matrix, maximize=True)
    dist = 0
    num_non_empty_surfaces = 0
    for surf_id_a, surf_id_b in zip(row_ind, col_ind):
        if not scene_a[surf_id_a].objects_on_surface and not scene_b[surf_id_b].objects_on_surface:
            continue
        # print(list_cats_a[surf_id_a])
        # print(list_cats_b[surf_id_b])
        # print("----")
        dist += 1 - sim_matrix[surf_id_a, surf_id_b]
        num_non_empty_surfaces += 1
    return dist/num_non_empty_surfaces
