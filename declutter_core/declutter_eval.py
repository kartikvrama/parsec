"""Functions for evaluating declutter model."""

from typing import Any, Dict, Union
from pathlib import Path
import torch

from declutter_core.encoder_decoder_model import DeclutterEncoderDecoder
from declutter_core.json_to_tensor import DeclutterBatchGen
from utils import utils_data
from utils import utils_eval

BATCH_SIZE = 1  # TODO: verify that multiple batches works.


def evaluate(
    model_params: Dict[str, Union[str, int, float]],
    checkpoint_path: Union[str, Path],
    batch_dict: Dict[str, Any],
    batch_generator: DeclutterBatchGen,
):
    """Evaluates the declutter model on a list of batched tensors."""
    result_array = []
    prediction_array = {}
    latent_embeddings = {}

    model = (
        DeclutterEncoderDecoder.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_params=model_params,
            batch_size=BATCH_SIZE,
            train_mode=False,
        )
        .double()
        .cuda()
    )
    model.eval()

    # Generate model outputs.
    print(f"Number of evaluation scenes: {len(batch_dict)}.")
    with torch.no_grad():
        struct_batch = {}
        for batch_num, (sid, example) in enumerate(batch_dict.items()):
            if batch_num % 100 == 0:
                print(f"Processing scene number {batch_num}: {sid}")
            struct_batch.update({sid: example})
            if len(struct_batch) < BATCH_SIZE:
                continue
            tensor_batch = batch_generator.batch_to_tensor(
                struct_batch, is_shuffle=False
            )
            response_generator = model.predict(tensor_batch)
            for response in response_generator:
                scene_id = response["scene_id"]
                scene_num = response["scene_num"]

                # Calculate evaluation metrics.
                predicted_scene = response["predicted_scene"]
                goal_scene = tensor_batch["goal_scene_list"][scene_num]
                edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                    predicted_scene, goal_scene
                )

                # Save evaluation metrics.
                unplaced_object_names = list(
                    o.name
                    for o in struct_batch[scene_id]["partial"][-1].objects_on_surface
                )
                num_misplaced_object_classes = len(set(unplaced_object_names))
                rdict = {
                    "scene_id": scene_id,
                    "container_type": struct_batch[scene_id]["container_type"],
                    "household": struct_batch[scene_id]["household"],
                    "num_demonstrations": struct_batch[scene_id]["num_demonstrations"],
                    "num_removed_demonstrations": sum(
                        int(x)
                        for x in struct_batch[scene_id]["num_removed_demonstrations"]
                    ),
                    "num_misplaced_objects": len(unplaced_object_names),
                    "num_misplaced_object_classes": num_misplaced_object_classes,
                    "edit_distance": edit_distance,
                    "igo": igo,
                    "ipc": ipc,
                }
                result_array.append(rdict)

                # Save predicted scenes.
                prediction_array[scene_id] = {
                    "demonstration_scenes": [
                        utils_data.scene_to_json(x)
                        for x in struct_batch[scene_id]["observed_arrangement_list"]
                    ],
                    "partial_scene": utils_data.scene_to_json(
                        struct_batch[scene_id]["partial"]
                    ),
                    "goal_scene": utils_data.scene_to_json(
                        struct_batch[scene_id]["goal"]
                    ),
                    "predicted_scene": utils_data.scene_to_json(predicted_scene),
                }

                # Save latent embeddings.
                latent_embeddings[scene_id] = {}
                for key in response:
                    if key not in ["scene_id", "scene_num", "predicted_scene"]:
                        latent_embeddings[scene_id][key] = response[key].cpu().numpy()
            struct_batch = {}  # Clear batch for next iteration.
    return result_array, prediction_array, latent_embeddings
