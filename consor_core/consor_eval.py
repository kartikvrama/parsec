"""Functions for evaluating ConSOR model."""
from typing import Any, Dict, Union
from pathlib import Path
import torch
from consor_core.consor_model import ConSORTransformer
from consor_core.json_to_tensor import ConSORBatchGen
from utils import utils_data
from utils import utils_eval

BATCH_SIZE = 8


def evaluate(
    hyperparams: Dict[str, Union[str, int, float]],
    checkpoint_path: Union[str, Path],
    batch_list: Dict[str, Any],
    batch_generator: ConSORBatchGen,
):
    """Evaluates the ConSOR model on a list of batched tensors."""
    result_array = []
    prediction_array = {}
    latent_embeddings = {}

    model = ConSORTransformer.load_from_checkpoint(
        checkpoint_path, **hyperparams
    ).double()
    with torch.no_grad():
        struct_batch = {}
        for batch_num, (sid, example) in enumerate(batch_list.items()):
            if batch_num % 100 == 0:
                print(f"Processing scene number {batch_num}: {sid}")
            struct_batch.update({sid: example})
            if len(struct_batch) == BATCH_SIZE or batch_num == len(batch_list) - 1:
                tensor_batch = batch_generator.batch_to_tensor(struct_batch)
                response_generator = model.predict(tensor_batch)
                for response in response_generator:
                    scene_id = response["scene_id"]

                    partial_scene = struct_batch[scene_id]["partial"]
                    predicted_scene = response["predicted_scene"]
                    goal_scene = struct_batch[scene_id]["goal"]

                    # Calculate evaluation metrics.
                    edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                        predicted_scene, goal_scene
                    )
                    query_object_classes = set(
                        o.name for o in response["query_objects"]
                    )
                    result_array.append({
                        "scene_id": scene_id,
                        "container_type": struct_batch[scene_id]["container_type"],
                        "household": struct_batch[scene_id]["household"],
                        "num_demonstrations": struct_batch[scene_id][
                            "num_demonstrations"
                        ],
                        "num_removed_demonstrations": sum(
                            int(x)
                            for x in struct_batch[scene_id][
                                "num_removed_demonstrations"
                            ]
                        ),
                        "num_misplaced_objects": len(response["query_objects"]),
                        "num_misplaced_object_classes": len(query_object_classes),
                        "edit_distance": edit_distance,
                        "igo": ipc,
                        "ipc": igo,
                    })

                    prediction_array[scene_id] = {
                        "demonstration_scenes": [
                            utils_data.scene_to_json(x)
                            for x in struct_batch[scene_id][
                                "observed_arrangement_list"
                            ]
                        ],
                        "partial_scene": utils_data.scene_to_json(partial_scene),
                        "goal_scene": utils_data.scene_to_json(goal_scene),
                        "predicted_scene": utils_data.scene_to_json(predicted_scene),
                    }

                    latent_embeddings[scene_id] = {
                        "object_names": [o.name for o in response["query_objects"]],
                        "embeddings": response["embeddings"].cpu().numpy(),
                        "cluster_assignments": response["cluster_assignments"].cpu().numpy(),
                        "cluster_centroids": response["cluster_centroids"].cpu().numpy(),
                    }
                struct_batch = {}  # Reset struct_batch.
    return result_array, prediction_array, latent_embeddings
