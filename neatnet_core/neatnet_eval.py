"""Function to perform evaluation of the model on the test set."""
from typing import Any, Dict, Union
from pathlib import Path
import torch
from neatnet_core.data_loader import NeatNetDataset
from neatnet_core.neatnet_model import NeatGraph
from neatnet_core import utils_neatnet
from utils import utils_data
from utils import utils_eval


def evaluate(
    hyperparams: Dict[str, Any],
    batch_generator: NeatNetDataset,
    checkpoint_path: Union[str, Path],
):
    """Loads model from checkpoint and performs evaluation on a given set of batches.
    """
    result_array = []
    prediction_array = {}
    latent_embeddings = {}

    model = NeatGraph.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hyperparams=hyperparams,
        train_mode=False,
    ).double().cuda()
    model.eval()

    batch_indices = list(range(len(batch_generator)))
    with torch.no_grad():
        # TODO: extend evaluation to multiple batches.
        for batch_num in batch_indices:
            key = batch_generator.key_list[batch_num]
            struct_batch = batch_generator.data_dict[key]
            tensor_batch = batch_generator.return_graph_batch(batch_num)
            if batch_num % 25 == 0:
                print(f"Processing scene number {batch_num}: {key}")
            (
                predicted_scene,
                pred_positions,
                u_mu,
                u_log_var,
                user_prefs
            ) = model.predict(tensor_batch)
            # TODO: compute true object positions in batch generator and
            # then calculate validation loss.
            goal_scene = tensor_batch["eval_pred_scene"]
            edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                predicted_scene, goal_scene
            )

            unplaced_objects = struct_batch["partial"][-1].objects_on_surface
            unplaced_object_classes = set(o.name for o in unplaced_objects)
            result_array.append({
                "scene_id": key,
                "container_type": struct_batch["container_type"],
                "household": struct_batch["household"],
                "num_demonstrations": struct_batch["num_demonstrations"],
                "num_removed_demonstrations": sum(
                    int(x) for x in struct_batch["num_removed_demonstrations"]
                ),
                "num_misplaced_objects": len(unplaced_objects),
                "num_misplaced_object_classes": len(unplaced_object_classes),
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
            })
            prediction_array[key] = {
                "predicted_scene": utils_data.scene_to_json(predicted_scene),
                "goal_scene": utils_data.scene_to_json(goal_scene),
            }
            latent_embeddings[key] = {
                "u_mu": u_mu.cpu().numpy(),
                "u_log_var": u_log_var.cpu().numpy(),
                "user_prefs": user_prefs.cpu().numpy(),
            }
    return result_array, prediction_array, latent_embeddings
