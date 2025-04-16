"""CF training functions."""
from typing import Tuple
import torch
import numpy as np
from cf_core.cf_model import CFModel

MAX_EPOCHS = 5000


def train_batch(
    model: CFModel,
    matrix_train: torch.Tensor,
    non_neg_indices_train: Tuple[np.ndarray, np.ndarray],
    learning_rate: float,
    convergence_threshold: float = 1e-3,
):
    """Train the CF model using L-BFGS optimizer and return train loss history.
    
    Args:
        model: CF model object.
        learning_rate: Learning rate for the optimizer.
        convergence_threshold: Threshold for convergence. Defaults to 1e-3.
    """
    # Optimizer and closure function.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    def _closure():
        optimizer.zero_grad()
        ratings_pred = model.forward()
        _, total_loss = model.calculate_loss(
            ratings_pred, matrix_train, non_neg_indices_train
        )
        total_loss.backward()
        return total_loss

    ep = 1
    loss_arr_train = []
    while True:
        loss_total_train = optimizer.step(_closure)
        if torch.isnan(loss_total_train):
            raise ValueError("Loss is NaN. Training failed.")
        loss_arr_train.append(loss_total_train.detach().cpu().numpy())
        # Calculate convergence rate.
        if len(loss_arr_train) >= 2:
            convergence_rate = (
                loss_arr_train[-2] - loss_arr_train[-1]
            ) / loss_arr_train[-2]
        else:
            convergence_rate = 1
        print(
            f"Epoch: {ep+1}, Train total loss: {loss_total_train}, Convergence: {convergence_rate}"
        )
        # End training if convergence rate is less than 1e-3.
        if convergence_rate < convergence_threshold:
            print(f"Training stopped at epoch {ep}.")
            break
        ep += 1
        if ep >= MAX_EPOCHS:
            print(f"Training stopped due to max epoch limit of {MAX_EPOCHS}.")
            break
    return loss_arr_train
