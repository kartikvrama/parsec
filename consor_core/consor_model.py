"""ConSOR Transformer Encoder model from Ramachandruni et al. 2023."""
from typing import Literal
from copy import deepcopy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from pytorch_metric_learning.reducers import DoNothingReducer
from pytorch_metric_learning.losses import TripletMarginLoss, NPairsLoss
import pytorch_lightning as pl

from utils import utils_data
from utils import utils_eval


class ConSORTransformer(pl.LightningModule):
    """ConSOR Transformer Model to infer goal arrangement from a partially
    arranged initial scene."""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_size: int,
        output_dimension: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        loss_fn: Literal["triplet_margin", "npairs"],
        train_batch_size: int,
        val_batch_size: int,
        lrate: float,
        wt_decay: float,
        triplet_loss_margin: float,
        train_mode: bool=False,
    ):
        """Initialize the transformer encoder."""

        super().__init__()

        # dimensionality of node features

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="relu",
            dim_feedforward=hidden_layer_size,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers
        )

        # Linear layer
        self.dropout_linear = nn.Dropout(0)
        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.01)
        self.linear = nn.Linear(input_dim, output_dimension)

        # Do not reduce loss to mean
        reducer = DoNothingReducer()

        # Contrastive loss function
        if loss_fn == "triplet_margin":  # triple margin loss
            self.contrastive_loss = TripletMarginLoss(
                margin=triplet_loss_margin, reducer=reducer
            )

        elif loss_fn == "npairs":  # npairs loss
            self.contrastive_loss = NPairsLoss(reducer=reducer)

        else:
            raise NotImplementedError(f"{loss_fn} is not implemented")

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # optimizer hyperparameters.
        self.optimizer_params = dict({"lrate": lrate, "wtdecay": wt_decay})

    def forward(self, input, padding_mask):
        """Forward pass of the transformer model.

        Args:
            input_tensor: Input tensor of shape batch_size x num_columns x d_node.
            padding_mask: Padding mask of shape
                batch_size x num_columns x d_node.

        Returns:
            Output tensor of shape batch_size x num_columns x d_output.
        """

        batch_len = input.size()[0]

        output = self.encoder(input, src_key_padding_mask=padding_mask)

        # Dimension reduction of transformer embeddings.
        output = self.linear(
            self.dropout_linear(self.activation(output.view(-1, output.size()[-1])))
        )
        output = output.view(batch_len, -1, output.size()[-1])

        output = F.normalize(output, p=2.0, dim=-1)
        return output

    def training_step(self, batch, batch_idx):
        """Performs one training step of the transformer model."""

        del batch_idx

        node_split = batch["node_split"]
        true_cluster_ids = batch["true_node_cluster_ids"]

        # edge_adj_pred is N x N output, N is num of nodes in batch graph.
        node_embeddings = self.forward(batch["input"], batch["padding_mask"])

        start = 0
        loss_array = None
        for scene_num, node_len in enumerate(node_split):
            node_embeddings_sample = node_embeddings[scene_num, :node_len, :]
            true_cluster_ids_scene = true_cluster_ids[scene_num]

            # very special case where there is only one container.
            if torch.unique(true_cluster_ids_scene).size()[0] == 1:
                loss_sample = torch.zeros(
                    (1,), dtype=torch.double, requires_grad=True
                ).to("cuda")
            else:
                # Unreduced loss from function.
                loss_dict = self.contrastive_loss(
                    node_embeddings_sample, true_cluster_ids_scene
                )

                loss_sample = loss_dict["loss"]["losses"]

            if isinstance(loss_sample, int):
                loss_sample = (0*node_embeddings_sample.ravel()[0]).unsqueeze(0)
            if loss_array is None:
                loss_array = loss_sample.clone()
            else:
                loss_array = torch.cat([loss_array, loss_sample])

            start += node_len

        # Non zero loss reduction.
        nonzero_loss_indices = torch.nonzero(loss_array)

        if len(nonzero_loss_indices) > 0:
            loss = torch.mean(loss_array[nonzero_loss_indices])

        else:  # all losses are zero, no non zero losses.
            loss = torch.sum(node_embeddings_sample * 0)

        # Record training loss.
        self.log(
            "train_loss",
            loss,
            batch_size=self.train_batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def predict(self, batch):
        """Predicts object placements from the partial scene and yields final arrangement."""
        pred_embeddings = self.forward(batch["input"], batch["padding_mask"])

        # Unpack batch.
        node_split = batch["node_split"]
        partial_scene_cluster_id_list = batch["partial_scene_cluster_ids"]
        object_list = batch["object_list"]
        struct_batch = batch["struct_batch"]
        surface_list = batch["surface_list"]

        # Unravel the batch into individual graphs.
        start = 0
        for scene_num, node_len in enumerate(node_split):
            # Unraveling the batch.
            pred_embeddings_scene = pred_embeddings[scene_num, :node_len, :]
            partial_scene_clusters = partial_scene_cluster_id_list[scene_num]
            scene_surfaces = surface_list[scene_num]

            # list of graph objects.
            scene_objects = object_list[scene_num]

            # Indexes of arranged objects/clusters and query objects.
            arranged_object_indices = torch.nonzero(partial_scene_clusters > 0).squeeze(
                1
            )
            query_object_indices = torch.nonzero(partial_scene_clusters == 0).squeeze(1)

            # Separate unplaced (query) objects.
            query_objects = list(scene_objects[i] for i in query_object_indices)

            all_cluster_indices = torch.unique(
                partial_scene_clusters[arranged_object_indices]
            )

            # Initialize cluster means.
            cluster_means = torch.empty(size=(0, pred_embeddings_scene.size()[1])).type(
                pred_embeddings_scene.type()
            )

            for cidx in all_cluster_indices:
                object_indices = torch.nonzero(partial_scene_clusters == cidx)
                mean_embb = torch.mean(pred_embeddings_scene[object_indices], dim=0)

                cluster_means = torch.cat([cluster_means, mean_embb], dim=0)

            query_node_embeddings = pred_embeddings_scene[query_object_indices]

            # Pairwise similarity matrix between query nodes and cluster means.
            query_mean_distances = torch.matmul(
                query_node_embeddings, cluster_means.t()
            )

            # Choose cluster with maximum similarity.
            pred_clusters_query_indices = torch.argmax(
                query_mean_distances, dim=1
            ).cpu()
            pred_clusters_query = pred_clusters_query_indices.apply_(
                lambda x: all_cluster_indices[x]
            )

            # Tuple of (key, scene).
            key, scene_dict = struct_batch[scene_num]

            # Place objects in scene.
            object_id_surface_tuples = list(
                [o.object_id, scene_surfaces[pred_clusters_query[i] - 1]]
                for i, o in enumerate([
                    scene_objects[i] for i in query_object_indices
                ])
            )  # Cluster IDs are 1-indexed, hence cluster ID - 1 = surface index.
            scene_pred = utils_eval.place_objects_in_scene(
                scene_dict["partial"], object_id_surface_tuples
            )

            start += node_len

            yield {
                "scene_num": scene_num,
                "scene_id": key,
                "predicted_scene": scene_pred,
                "embeddings": pred_embeddings_scene,
                "cluster_assignments": partial_scene_clusters,
                "cluster_centroids": cluster_means,
                "query_objects": query_objects,
            }

    def validation_step(self, batch, batch_idx):
        """Performs evaluation on the (full) validation data."""

        del batch_idx

        struct_batch = batch["struct_batch"]
        response_generator = self.predict(batch)
        result_array = []
        for response in response_generator:
            scene_num = response["scene_num"]

            predicted_scene = response["predicted_scene"]
            goal_scene = struct_batch[scene_num][1]["goal"]

            # Calculate evaluation metrics.
            edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                predicted_scene, goal_scene
            )
            result_array.append(
                {
                    "edit_distance": edit_distance,
                    "ipc": ipc,
                    "igo": igo,
                }
            )

        # Record validation success rate and NED averaged across validation batches.
        success_rate = (
            1.0
            * len([x["edit_distance"] for x in result_array if x["edit_distance"] == 0])
            / len(result_array)
        )
        self.log(
            "success_rate",
            float(success_rate),
            batch_size=len(struct_batch),
            prog_bar=True,
        )
        if (
            len([x["edit_distance"] for x in result_array if x["edit_distance"] > 0])
            > 0
        ):
            ned = (
                1.0
                * sum(
                    x["edit_distance"] for x in result_array if x["edit_distance"] > 0
                )
                / len(
                    [x["edit_distance"] for x in result_array if x["edit_distance"] > 0]
                )
            )
        else:
            ned = 0
        self.log(
            "non_zero_edit_distance",
            float(ned),
            batch_size=len(result_array),
            prog_bar=True,
        )
        self.log(
            "edit_distance",
            float(
                sum(list(x["edit_distance"] for x in result_array))
                / len(list(x["edit_distance"] for x in result_array))
            ),
            batch_size=len(result_array),
            prog_bar=True,
        )
        self.log(
            "ipc",
            float(
                sum(list(x["ipc"] for x in result_array))
                / len(list(x["ipc"] for x in result_array))
            ),
            batch_size=len(result_array),
            prog_bar=True,
        )
        self.log(
            "igo",
            float(
                sum(list(x["igo"] for x in result_array))
                / len(list(x["igo"] for x in result_array))
            ),
            batch_size=len(result_array),
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Configures the pytorch lightning optimizer."""
        return Adam(
            self.parameters(),
            lr=self.optimizer_params["lrate"],
            weight_decay=self.optimizer_params["wtdecay"],
        )
