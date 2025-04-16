"""Transformer encoder model to infer object placement from partial initial scene."""
from typing import Dict, Literal, Union, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_metric_learning import losses, reducers
from pytorch_lightning.utilities import grad_norm
import wandb

from utils import utils_eval
from utils import utils_torch
from declutter_core import utils_declutter

# TODO: convert to hyperparameters.
EMBEDDING_DIM = 32
DEMO_POS_DIM = 32

def _padded_batch_tensor_to_list(
        input_tensor: torch.Tensor, split_mask: torch.Tensor):
    """Slices a padded tensor to a list of unpadded tensors.

    The function assumes padding is along dimension 1, and considers unequally
    padding along the padded dimension. It uses the mask to slice the padded
    tensor.
    """
    assert len(input_tensor.size()) == 3
    assert input_tensor.size()[:2] == split_mask.size()
    batch_len = input_tensor.size()[0]
    input_tensor_list = []
    for i in range(batch_len):
        indices = torch.nonzero(split_mask[i, :] == 1).squeeze()
        sliced_tensor = input_tensor[i, indices, :]
        if len(sliced_tensor.size()) == 1:
            sliced_tensor = sliced_tensor.unsqueeze(0)
        input_tensor_list.append(sliced_tensor)
    return input_tensor_list


class DeclutterEncoderDecoder(pl.LightningModule):
    """Declutter Transformer Model to infer goal arrangement from a partially
    arranged initial scene and user demonstration(s)."""

    def __init__(
        self,
        model_params: Dict[str, Union[float, int, str]],
        batch_size: int=None,
        lrate: float=None,
        wt_decay: float=None,
        alpha: float=1.0,
        beta: float=1.0,
        triplet_margin_main: float=0.75,
        triplet_margin_aux: float=0.75,
        lr_scheduler_tmax: Optional[float]=None,
        mode: Literal["train", "eval"] = "train",
    ):
        """Initialize the transformer model."""

        super().__init__()

        # Object, surface and type embedding dimensions of node features.
        self.seq_tensor_dim = (
            model_params["object_dimension"]
            + EMBEDDING_DIM
            + model_params["surface_grid_dimension"]
            + model_params["instance_encoder_dim"]
            + DEMO_POS_DIM
        )
        self.scene_tensor_dim = (
            model_params["object_dimension"]
            + EMBEDDING_DIM
            + model_params["surface_grid_dimension"]
            + model_params["instance_encoder_dim"]
        )

        # Define surface type learned embeddings.
        self.generate_surface_embedding = torch.nn.Embedding(
            model_params["num_surface_types"], EMBEDDING_DIM
        )

        # Define instance encoders.
        self.generate_object_encoding = utils_torch.PositionalEncodingCaller(
            max_position=32, d_model=model_params["instance_encoder_dim"]
        )
        self.generate_demo_encoding = utils_torch.PositionalEncodingCaller(
            max_position=16, d_model=DEMO_POS_DIM
        )

        self.relu_activation = nn.ReLU()
        self.hidden_layer_size = model_params["hidden_layer_size"]

        # Decoder for encoding demonstration sequence.
        sequence_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.seq_tensor_dim,
            nhead=model_params["num_heads"],
            activation="relu",
            dim_feedforward=self.hidden_layer_size,
            dropout=model_params["dropout"],
            batch_first=True,
            norm_first=False,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=sequence_encoder_layer,
            num_layers=model_params["num_layers"]
        )
        # Projection layer for sequence latents.
        self.sequence_projection = nn.Sequential(
            nn.Linear(self.seq_tensor_dim, self.scene_tensor_dim),
            nn.ReLU(),
        )

        # Object decoder (conditioned by output of demonstration encoder).
        object_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.scene_tensor_dim,
            nhead=model_params["num_heads"],
            activation="relu",
            dim_feedforward=self.hidden_layer_size,
            dropout=model_params["dropout"],
            batch_first=True,
            norm_first=False,
        )
        self.object_decoder = nn.TransformerDecoder(
            decoder_layer=object_decoder_layer,
            num_layers=model_params["num_layers"]
        )

        # Losses and loss constants.
        reducer = reducers.DoNothingReducer()  # Triplet margin loss function.
        self.margin_loss_main = losses.TripletMarginLoss(
            margin=triplet_margin_main, reducer=reducer
        )
        self.margin_loss_aux = losses.TripletMarginLoss(
            margin=triplet_margin_aux, reducer=reducer
        )
        self.alpha = alpha
        self.beta = beta
        print(f"Alpha: {self.alpha}, Beta: {self.beta}")
        print(f"Triplet margin main: {triplet_margin_main}, Triplet margin aux: {triplet_margin_aux}")

        # Train and optimizer hyperparameters.
        self.batch_size = batch_size
        self.mode = mode
        if self.mode == "train":
            self.optimizer_params = dict(
                {
                    "lrate": lrate,
                    "wtdecay": wt_decay,
                    "lr_scheduler_tmax": lr_scheduler_tmax
                }
            )

    def _batch_to_forward_pass(self, batch):
        return (
            batch["seq_ostensor"], batch["seq_demo_index"],
            batch["seq_split_mask"], batch["seq_padding_mask"],
            batch["container_type_tensor"],
            batch["scene_ostensor"], batch["scene_padding_mask"],
            batch["partial_scene_cluster_mask"]
        )

    def _calculate_latent_centroid(
        self, aggregate_latent: torch.Tensor, num_scenes_per_cluster: torch.Tensor
    ) -> torch.Tensor:
        """Computes the centroid of the aggregate latents.
        """
        num_scenes_per_cluster = torch.where(
            num_scenes_per_cluster > 0, num_scenes_per_cluster, torch.ones_like(num_scenes_per_cluster)
        )
        return aggregate_latent / num_scenes_per_cluster

    def _encode_sequence(self, latent, demo_index, split_mask, padding_mask):
        """Combines demonstrations into a shared latent space."""
        demo_index_embb = self.generate_demo_encoding(demo_index)
        if len(demo_index_embb.size()) == 2:
            demo_index_embb = demo_index_embb.unsqueeze(0)
        latent = torch.mul(
            torch.cat([latent, demo_index_embb], dim=-1),
            torch.mul(split_mask, 1 - padding_mask).unsqueeze(-1)
        )
        return self.sequence_encoder(
            src=latent, src_key_padding_mask=padding_mask
        )

    def _obj_surf_tensor_to_latent(
        self, tensor: utils_declutter.ObjectSurfaceTensor, padding_mask: torch.tensor
    ):
        """Converts object and surface tensors to a single latent tensor."""
        surface_latent = self.generate_surface_embedding(
            tensor.surface_type.reshape(-1)
        ).reshape(tensor.surface_type.size()[:2] + (-1,))
        object_index_embb = self.generate_object_encoding(
            torch.tensor(tensor.object_copy_index, device=self.device, dtype=torch.long)
        )
        latent = torch.cat([
            tensor.object_class, surface_latent, tensor.grid, object_index_embb
        ], dim=-1)
        return torch.mul(latent, 1 - padding_mask.unsqueeze(-1))

    def forward(self, batch):
        """Forward pass of the transformer model.
        """
        (
            seq_ostensor, seq_demo_index, seq_split_mask,
            seq_padding_mask, _, scene_ostensor, scene_padding_mask, _
        ) = self._batch_to_forward_pass(batch)

        # A. Encode demonstration sequence tensor.
        seq_latent = self._obj_surf_tensor_to_latent(seq_ostensor, seq_padding_mask)
        seq_latent = self._encode_sequence(
            seq_latent, seq_demo_index, seq_split_mask, seq_padding_mask
        )
        seq_latent_reduced = self.sequence_projection(
            seq_latent.reshape(-1, seq_latent.size()[-1])
        ).reshape(seq_latent.size()[:2] + (-1,))

        # B. Encode combined object tensor in a shared latent space.
        scene_latent = self._obj_surf_tensor_to_latent(scene_ostensor, scene_padding_mask)
        scene_latent = self.object_decoder(
            tgt=scene_latent, memory=seq_latent_reduced,
            tgt_key_padding_mask=scene_padding_mask,
            memory_key_padding_mask=seq_padding_mask
        )

        return (
            F.normalize(seq_latent_reduced, p=2, dim=-1),
            F.normalize(scene_latent, p=2, dim=-1)
        )

    def predict(self, batch):
        """Predicts the goal scene from the partial scene and demonstration sequence.
        """
        batch_len = batch["seq_demo_index"].size()[0]
        _, scene_latent = self.forward(batch)
        # Process model outputs.
        placed_mask = torch.mul(
            batch["partial_scene_split_mask"], 1 - batch["scene_padding_mask"]
        )
        unplaced_mask = torch.mul(
            1 - batch["partial_scene_split_mask"], 1 - batch["scene_padding_mask"]
        )
        for scene_num in range(batch_len):
            unplaced_latent_snum = scene_latent[scene_num][
                torch.nonzero(unplaced_mask[scene_num]).squeeze(), :
            ]
            placed_nonzero_indices = torch.nonzero(placed_mask[scene_num]).squeeze()
            placed_latent_snum = scene_latent[scene_num][
                torch.nonzero(placed_mask[scene_num]).squeeze(), :
            ]
            placed_cluster_matrix = batch["partial_scene_cluster_mask"][scene_num][
                placed_nonzero_indices, :
            ]  # Only possible because placed objects always come first in scene_latent.
            placed_cluster_matrix = placed_cluster_matrix[
                :, torch.where(placed_cluster_matrix.sum(dim=0) > 0)[0]
            ]
            assert placed_cluster_matrix.size()[1] == len(batch["partial_scene_list"][scene_num]) - 1
            placed_cluster_aggregate = torch.einsum(
                "ij,ik->jk", placed_cluster_matrix, placed_latent_snum
            )
            num_objects_per_cluster = torch.sum(placed_cluster_matrix, dim=0, keepdim=True).T
            cluster_centroids = placed_cluster_aggregate / num_objects_per_cluster
            assert not torch.isnan(cluster_centroids).all()

            partial_scene_struct = batch["partial_scene_list"][scene_num]
            surface_list = [s.name for s in partial_scene_struct[:-1]]

            predicted_object_id_placements = []
            for obj, latent in zip(
                batch["unplaced_object_ids"][scene_num], unplaced_latent_snum
            ):
                placement = torch.argmin(
                    torch.sum((cluster_centroids - latent.unsqueeze(0))**2, dim=1)
                )
                predicted_object_id_placements.append((obj, surface_list[placement]))

            predicted_scene_struct = utils_eval.place_objects_in_scene(
                scene_struct=partial_scene_struct,
                object_id_surface_tuples=predicted_object_id_placements
            )
            yield {
                "scene_id": batch["scene_ids"][scene_num],
                "scene_num": scene_num,
                "predicted_scene": predicted_scene_struct,
                "placed_latent": placed_latent_snum,
                "unplaced_latent": unplaced_latent_snum,
                "cluster_centroids": cluster_centroids,
            }

    def training_step(self, batch, batch_idx):
        """Performs one training step of the transformer encoder.

        Args:
            batch: Input batch. See return_tensor_batch method of
                declutter_core.data-loader module.
            batch_idx: Default pytorch lightning argument.
        """
        if batch is None:
            return torch.tensor(0, device=self.device)
        del batch_idx
        batch_len = batch["seq_demo_index"].size()[0]

        seq_latent, scene_latent = self.forward(batch)
        main_loss_array = None
        aux_loss_array = None
        for scene_num in range(batch_len):
            # --- Main metric loss on assigning all objects to surfaces. ---
            scene_latent_unpadded = scene_latent[scene_num][
                torch.where(batch["scene_padding_mask"][scene_num] == 0)[0], :
            ]
            main_loss_sample = self.margin_loss_main(
                scene_latent_unpadded, batch["object_triplet_labels"][scene_num].to(torch.long)
            )["loss"]["losses"]
            if isinstance(main_loss_sample, int):
                # When no two objects in the sample belong to the same cluster.
                main_loss_sample = (0*scene_latent.ravel()[0]).unsqueeze(0)
            if main_loss_array is None:
                main_loss_array = main_loss_sample.clone()
            else:
                main_loss_array = torch.cat([main_loss_array, main_loss_sample])

            # --- Auxilliary loss to cluster demonstration latents. ---
            seq_latent_unpadded = seq_latent[scene_num][
                torch.where(batch["seq_padding_mask"][scene_num] == 0)[0], :
            ]  # Removing sequence padding.
            end_token_positions = torch.where(batch["seq_split_mask"][scene_num] == 0)[0]
            if len(end_token_positions) == 0:
                tuples = [(seq_latent_unpadded, batch["seq_cluster_labels"][scene_num])]
            else:
                seq_labels = batch["seq_cluster_labels"][scene_num]
                tuples = []
                prev_pos = 0
                for token_id, pos in enumerate(end_token_positions):
                    if token_id == 0:
                        seq_chunk = seq_latent_unpadded[:pos, :]
                    else:
                        seq_chunk = seq_latent_unpadded[end_token_positions[token_id-1]+1:pos, :]
                    label_chunk = seq_labels[prev_pos:prev_pos + len(seq_chunk)]
                    tuples.append((seq_chunk, label_chunk))
                    prev_pos += len(seq_chunk)
                assert len(seq_latent_unpadded) - (end_token_positions[-1] + 1) == len(seq_labels) - prev_pos, (
                    f"Length mismatch between sequence latents and labels.: {len(seq_latent_unpadded) - (end_token_positions[-1] + 1)} vs {len(seq_labels) - prev_pos}"
                )
                tuples.append(
                    (
                        seq_latent_unpadded[end_token_positions[-1]+1:, :],
                        seq_labels[prev_pos:]
                    )
                )
            for tup in tuples:
                aux_loss_sample = self.margin_loss_aux(tup[0], tup[1])["loss"]["losses"]
                if isinstance(aux_loss_sample, int):
                    # When no two objects in the sample belong to the same cluster.
                    aux_loss_sample = (0*seq_latent.ravel()[0]).unsqueeze(0)
                if aux_loss_array is None:
                    aux_loss_array = aux_loss_sample.clone()
                else:
                    aux_loss_array = torch.cat([aux_loss_array, aux_loss_sample])

        main_nonzero_indices = torch.nonzero(main_loss_array).squeeze()
        if main_nonzero_indices.numel() > 0:
            main_loss = torch.mean(main_loss_array[main_nonzero_indices])
        else:
            main_loss = 0*scene_latent.ravel()[0]
        aux_nonzero_indices = torch.nonzero(aux_loss_array).squeeze()
        if aux_nonzero_indices.numel() > 0:
            aux_loss = torch.mean(aux_loss_array[aux_nonzero_indices])
        else:
            aux_loss = 0*seq_latent.ravel()[0]
        total_loss = self.alpha*main_loss + self.beta*aux_loss

        # Record training loss.
        self.log(
            "main_cluster_loss",
            main_loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "main_cluster_loss_nonzero",
            torch.mean(main_loss_array),
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "aux_cluster_loss",
            aux_loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "aux_cluster_loss_nonzero",
            torch.mean(aux_loss_array),
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "total_loss",
            total_loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Performs one validation step of the transformer encoder.

        Args:
            batch: Input batch. See generate_tensor_batch method of
                declutter data loader.
            batch_idx: Default pytorch lightning argument.
        """
        if batch is None:
            return
        del batch_idx
        if self.trainer.global_step == 0:
            wandb.define_metric("success_rate", summary="mean")
            wandb.define_metric("edit_distance", summary="mean")
            wandb.define_metric("igo", summary="mean")
            wandb.define_metric("ipc", summary="mean")
        batch_len = batch["seq_demo_index"].size()[0]

        edit_distance_array = []
        ipc_array = []
        igo_array = []
        response_generator = self.predict(batch)
        for response in response_generator:
            predicted_scene = response["predicted_scene"]
            goal_scene = batch["goal_scene_list"][response["scene_num"]]
            edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                predicted_scene, goal_scene
            )
            edit_distance_array.append(edit_distance)
            ipc_array.append(ipc)
            igo_array.append(igo)

        # Record validation success rate and NED averaged across validation batches.
        success_rate = 1.0*len([x for x in edit_distance_array if x == 0]) / len(
            edit_distance_array
        )
        self.log(
            "success_rate",
            float(success_rate),
            batch_size=batch_len,
            prog_bar=True,
        )
        self.log(
            "edit_distance",
            float(sum(edit_distance_array) / len(edit_distance_array)),
            batch_size=batch_len,
            prog_bar=True,
        )
        self.log(
            "ipc",
            float(sum(ipc_array) / len(ipc_array)),
            batch_size=batch_len,
            prog_bar=True,
        )
        self.log(
            "igo",
            float(sum(igo_array) / len(igo_array)),
            batch_size=batch_len,
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Configures the pytorch lightning optimizer."""

        optimizer = Adam(
            self.parameters(),
            lr=self.optimizer_params["lrate"],
            weight_decay=self.optimizer_params["wtdecay"],
        )
        if self.optimizer_params["lr_scheduler_tmax"] == "None":
            return optimizer
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.optimizer_params["lr_scheduler_tmax"],
            eta_min=1e-6, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = {}
        for layer in [
            self.sequence_encoder.layers, self.object_decoder.layers,
            self.sequence_projection, self.generate_surface_embedding
        ]:
            norms.update(grad_norm(layer, norm_type=2))
        self.log_dict(norms)
