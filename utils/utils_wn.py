"""Helper functions for clustering using WordNet labels"""

from typing import Any, List
import csv
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def load_category_labels():
    """Loads category labels from a pre-defined file."""
    # TODO: delete.
    with open("labels/object_to_category.csv", "r") as fcsv:
        data = csv.DictReader(fcsv)
        obj2cat = {row["name"]: row["category"] for row in data}
    return obj2cat


def load_wordnet_labels():
    """Loads wordnet labels from a pre-defined file."""
    with open("labels/object_to_wordnet.csv", "r") as fcsv:
        data = csv.DictReader(fcsv)
        obj2wn = {row["name"]: row["synset"] for row in data}
    return obj2wn


def return_pairwise_distance_matrix(synsets: List[Any]):
    """Returns a pairwise distance matrix between wordnet synsets"""
    dist_mat = np.zeros((len(synsets), len(synsets)))
    for i, sn_i in enumerate(synsets):
        for j, sn_j in enumerate(synsets):
            if i == j:
                continue
            dist_mat[i, j] = 1 - sn_i.wup_similarity(sn_j)
    return dist_mat


def lowest_common_hypernym(synsets):
    """
    Finds the lowest common hypernym (LCH) for a list of synsets.
    
    :param synsets: List of synsets
    :return: List of lowest common hypernyms (usually only one)
    """
    if not synsets:
        return None
    # Initialize the lowest common hypernym with the first synset
    lch = synsets[0]
    # Iteratively find the lowest common hypernym for the rest of the synsets
    for synset in synsets[1:]:
        lch = lch.lowest_common_hypernyms(synset)[0]
    return lch


def return_cluster_assignments(pairwise_distance_matrix: np.ndarray, threshold: float):
    """Performs hierarchical clustering and returns cluster labels."""
    clusters = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=threshold
    ).fit(pairwise_distance_matrix)
    return clusters.labels_


def return_cluster_parents(synsets: List[Any], cluster_labels: List[int]):
    """Assigns wordnet labels to clusters based on lowest common hypernym."""
    assert len(synsets) == len(cluster_labels)
    label_to_synset = []
    for l in range(min(cluster_labels), 1 + max(cluster_labels)):
        common_synset = lowest_common_hypernym(
            list(
                sn for sn, l_i in zip(synsets, cluster_labels)
                if l_i == l
            )
        )
        label_to_synset.append(common_synset)
    return label_to_synset
