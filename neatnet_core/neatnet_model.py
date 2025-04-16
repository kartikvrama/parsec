"""NeatNet model for imitating placement preferences."""

from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric import nn as geom_nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl

from neatnet_core import utils_neatnet
from utils import utils_data
from utils import utils_eval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeatGraph(pl.LightningModule):
    def __init__(
        self,
        hyperparams: Dict[str, Any],
        train_mode=False,
    ):
        super(NeatGraph, self).__init__()

        # TODO: Uncomment this if you want to use FastText embeddings.
        # # Word embedding layer, takes word index and returns word vector.
        # if hyperparams["we_model"] is not None:
        #     self.word_embedding = nn.Embedding.from_pretrained(
        #         hyperparams["we_model"], freeze=True
        #     )

        self.user_dim = hyperparams["user_dim"]
        self.pos_dim = hyperparams["pos_dim"]
        self.semantic_dim = hyperparams["semantic_dim"]
        self.relu_leak = hyperparams["relu_leak"]
        self.encoder_h_dim = hyperparams["encoder_h_dim"]
        self.predictor_h_dim = hyperparams["predictor_h_dim"]
        self.graph_dim = hyperparams["graph_dim"]
        self.vae_beta = hyperparams["vae_beta"]
        self.hyperparams = hyperparams  # TODO: unwrap all hyperparams.

        # Node encoder.
        self.semantic_extractor = nn.Sequential(
            # self.word_embedding,
            nn.Linear(self.semantic_dim, self.semantic_dim * 2),
            nn.LeakyReLU(negative_slope=self.relu_leak),
            nn.Linear(self.semantic_dim * 2, self.semantic_dim),
        )

        # Graph encoder.
        self.graph_encoder = geom_nn.Sequential(
            "x, edge_index",
            [
                (
                    geom_nn.GATConv(
                        self.pos_dim + self.semantic_dim, self.encoder_h_dim
                    ),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                nn.Linear(self.encoder_h_dim, self.graph_dim),
            ],
        )

        # User encoder.
        self.user_encoder = nn.Sequential(
            nn.Linear(self.graph_dim, 4 * self.user_dim),
            nn.LeakyReLU(negative_slope=self.relu_leak),
            nn.Linear(4 * self.user_dim, 2 * self.user_dim),
        )

        # Position predictor.
        self.pos_predictor = geom_nn.Sequential(
            "x, edge_index",
            [
                (
                    geom_nn.GATConv(
                        self.semantic_dim + self.user_dim, self.predictor_h_dim
                    ),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                nn.Linear(self.predictor_h_dim, 4 * self.pos_dim),
                nn.ELU(),
                nn.Linear(4 * self.pos_dim, self.pos_dim),
            ],
        )

        # For evaluation.
        _, self.surface_constants = utils_data.return_object_surface_constants()

        # Train and optimizer hyperparameters.
        self.train_mode = train_mode

    # Takes nodes which start with word index, returns nodes with semantic embedding there instead.
    # Keeps rest of node features (usually position encoding) intact.
    def embed_nodes_semantically(self, nodes):
        node_words = nodes[:, :self.semantic_dim]
        node_semantics = self.semantic_extractor(node_words)

        # Optionally you can normalise the semantic embedding here: in some cases, can speed up learning.
        # node_norms = node_semantics.norm(p=2, dim=1, keepdim=True)
        # normed_semantics = node_semantics.div(node_norms)
        normed_semantics = node_semantics

        embedded_nodes = torch.empty_like(nodes).to(DEVICE)
        embedded_nodes[:, :self.semantic_dim] = normed_semantics
        embedded_nodes[:, self.semantic_dim:] = nodes[:, self.semantic_dim:]
        return embedded_nodes

    # Returned matrices have shape (batch_size, user_dim).
    def encode_user_prefs(self, example_graphs, batch_ids, scene_ids):
        example_nodes, example_edges = example_graphs

        # Extract semantic embeddings.
        embedded_nodes = self.embed_nodes_semantically(example_nodes)
        node_reps = self.graph_encoder(embedded_nodes, example_edges)

        # We want a different vector for each scene_id within each batch_id
        unique_scene_ids = utils_neatnet.create_unique_scene_ids(batch_ids, scene_ids)

        graph_rep = geom_nn.global_add_pool(node_reps, unique_scene_ids)

        # Encoder estimates user preferences separately for each example from one user.
        encoded = self.user_encoder(graph_rep)
        mu, log_var = torch.split(encoded, [self.user_dim, self.user_dim], dim=-1)
        return mu, log_var

    # Input matrices have shape (batch_size, user_dim).
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn(std.size()).to(DEVICE)
            user_prefs = mu + eps * std
        else:
            # Take mean u vectors at inference time.
            user_prefs = mu

        return user_prefs

    # Matrix user_prefs has shape (batch_size, user_dim).
    # Returned matrix is node feature graph.
    def decode(self, user_prefs, pred_graphs, batch_ids, scene_ids):
        node_words, edge_index = pred_graphs
        semantic_features = self.embed_nodes_semantically(node_words)

        # For each batch element, stretch user_prefs so that all nodes with same
        # batch_id have same row of user_prefs.
        user_prefs = torch.index_select(user_prefs, 0, batch_ids)

        node_features = torch.cat([semantic_features, user_prefs], dim=-1)

        pos_preds = self.pos_predictor(node_features, edge_index)

        return pos_preds

    # batch_ids: assigns each node to an example id within the batch.
    def forward(
        self,
        example_graphs,
        pred_graphs,
        example_batch_ids,
        pred_batch_ids,
        example_scene_ids,
        pred_scene_ids,
    ):
        u_mu, u_log_var = self.encode_user_prefs(
            example_graphs, example_batch_ids, example_scene_ids
        )
        all_user_prefs = self.reparameterize(u_mu, u_log_var)

        # Average user prefs across scenes for same user.
        num_users = example_batch_ids[-1].item() + 1
        user_prefs = torch.zeros(num_users, self.user_dim).to(DEVICE)
        scenes_so_far = 0
        # Note: this user ID is a temporary ID that does not correspond to the
        # user ID in the dataset.
        for user_id in torch.arange(num_users):
            user_mask = example_batch_ids == user_id
            user_scene_ids = torch.masked_select(example_scene_ids, user_mask)

            # Get number of example scenes for current user.
            user_num_scenes = user_scene_ids[-1].item() + 1

            # Aggregate user prefs based on each scene for current user.
            encodings_for_user = all_user_prefs[
                scenes_so_far : scenes_so_far + user_num_scenes
            ]
            mean_user_prefs = torch.mean(encodings_for_user, 0)

            user_prefs[user_id] = mean_user_prefs
            scenes_so_far += user_num_scenes

        pos_preds = self.decode(user_prefs, pred_graphs, pred_batch_ids, pred_scene_ids)
        return pos_preds, u_mu, u_log_var, user_prefs

    def training_step(self, batch, batch_idx):
        """Performs one training step of NeatNet."""
        if "user_id" in batch:
            assert isinstance(batch["user_id"], str)
            batch_len = 1
        else:
            assert "user_id_list" in batch and isinstance(batch["user_id_list"], list)
            batch_len = len(batch["user_id_list"])

        example_nodes, example_edges, example_scene_ids, example_batch_ids = (
            batch["example_nodes"],
            batch["example_edges"],
            batch["example_scene_ids"],
            batch["example_batch_ids"],
        )
        example_graphs = (example_nodes, example_edges)
        pred_nodes, pred_edges, pred_scene_ids, pred_batch_ids = (
            batch["train_pred_nodes"],
            batch["train_pred_edges"],
            batch["train_pred_scene_ids"],
            batch["train_pred_batch_ids"],
        )
        pred_graphs = (pred_nodes, pred_edges)

        pos_preds, u_mu, u_log_var, user_prefs = self.forward(
            example_graphs,
            pred_graphs,
            example_batch_ids.to(torch.long),
            pred_batch_ids.to(torch.long),
            example_scene_ids,
            pred_scene_ids,
        )
        true_pos = batch["train_pred_positions"]
        loss_total, loss_pred, loss_kl = utils_neatnet.loss_vae(
            pos_preds, true_pos, u_mu, u_log_var, self.vae_beta
        )

        # Log losses.
        self.log(
            "total_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True,
            batch_size=batch_len)
        self.log(
            "pred_loss", loss_pred, on_step=True, on_epoch=True, prog_bar=False,
            batch_size=batch_len
        )
        self.log(
            "kl_div_loss", loss_kl, on_step=True, on_epoch=True, prog_bar=False,
            batch_size=batch_len
        )
        return loss_total

    def predict(self, batch):
        """Predicts object placements from observation scenes."""
        example_nodes, example_edges, example_scene_ids, example_batch_ids = (
            batch["example_nodes"],
            batch["example_edges"],
            batch["example_scene_ids"],
            batch["example_batch_ids"],
        )
        example_graphs = (example_nodes, example_edges)

        goal_nodes, goal_edges, goal_objects, eval_batch_ids = (
            batch["eval_pred_nodes"],
            batch["eval_pred_edges"],
            batch["eval_pred_object_entities"],
            batch["eval_pred_batch_ids"],
        )
        goal_scene_ids = torch.zeros(goal_nodes.size(0), dtype=torch.long).to(DEVICE)
        goal_graphs = (goal_nodes, goal_edges)

        pred_positions, u_mu, u_log_var, user_prefs = self.forward(
            example_graphs,
            goal_graphs,
            example_batch_ids.to(torch.long),
            eval_batch_ids.to(torch.long),
            example_scene_ids,
            goal_scene_ids,
        )
        mean_x, mean_y, norm_x, norm_y = (
            batch["mean_x"],
            batch["mean_y"],
            batch["norm_x"],
            batch["norm_y"],
        )

        # Denormalize predictions.
        pred_positions = (
            0.5 * pred_positions * torch.tensor([norm_x, norm_y], device=DEVICE)
            + torch.tensor([mean_x, mean_y], device=DEVICE)
        )

        surface_position_dict = batch["surface_position_dict"]
        surface_names = list(surface_position_dict.keys())
        surface_positions = torch.stack(
            [surface_position_dict[s] for s in surface_names]
        )

        # Find closest matching surface.
        pred_positions_mat = pred_positions.unsqueeze(1).expand(
            -1, surface_positions.size(0), -1
        )
        surface_positions_mat = surface_positions.unsqueeze(0).expand(
            pred_positions.size(0), -1, -1
        )
        distances = torch.norm(pred_positions_mat - surface_positions_mat, dim=-1)
        closest_surface_indices = torch.argmin(distances, dim=-1)
        predicted_object_entity_placements = list(
            (obj, surface_names[closest_surface_indices[i].item()])
            for i, obj in enumerate(goal_objects)
        )

        # Compare predicted and true placements.
        if self.train_mode:
            predicted_scene = utils_eval.object_placements_to_scene_struct(
                predicted_object_entity_placements,
                surface_names,
                self.surface_constants
            )
        else:
            predicted_object_id_placements = list(
                (obj.object_id, sname)
                for obj, sname in predicted_object_entity_placements
            )
            partial_scene = batch["eval_partial_scene"]
            predicted_scene = utils_eval.place_objects_in_scene(
                partial_scene, predicted_object_id_placements
            )
        return predicted_scene, pred_positions, u_mu, u_log_var, user_prefs

    def validation_step(self, batch, batch_idx):
        """Performs one validation step of NeatNet."""
        # TODO: modify to handle batch size > 1.
        if "user_id" in batch:
            assert isinstance(batch["user_id"], str)
            batch_len = 1
        else:
            assert "user_id_list" in batch and isinstance(batch["user_id_list"], list)
            batch_len = len(batch["user_id_list"])
        if batch_len > 1:
            raise ValueError("Batch size > 1 not supported.")

        predicted_scene, pred_positions, u_mu, u_log_var, _ = self.predict(batch)
        true_pred_positions = batch["eval_pred_positions"]
        # Loss metrics.
        val_loss_total, val_loss_pred, val_loss_kl = utils_neatnet.loss_vae(
            pred_positions, true_pred_positions, u_mu, u_log_var, self.vae_beta
        )
        # Evaluation metrics.
        goal_scene = batch["eval_pred_scene"]
        edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
            predicted_scene, goal_scene
        )

        # Log validation losses.
        self.log(
            "val_total_loss",
            val_loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_len,
        )
        self.log(
            "val_pred_loss",
            val_loss_pred,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_len,
        )
        self.log(
            "val_kl_div_loss",
            val_loss_kl,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_len,
        )

        # TODO: convert to success rate, nzed and average ipc, igo.
        # Log evaluation metrics on validation set.
        self.log(
            "edit_distance",
            edit_distance,
            batch_size=batch_len,
            prog_bar=False,
        )
        self.log(
            "ipc",
            ipc,
            batch_size=batch_len,
            prog_bar=False,
        )
        self.log(
            "igo",
            igo,
            batch_size=batch_len,
            prog_bar=False,
        )

    def configure_optimizers(self):
        """Configures the pytorch lightning optimizer."""
        optimizer = SGD(
            self.parameters(),
            lr=self.hyperparams["init_lr"],
            momentum=0.9,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.hyperparams["sch_factor"],
            patience=self.hyperparams["sch_patience"],
            threshold=0.0001,
            cooldown=self.hyperparams["sch_cooldown"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
            },
        }
