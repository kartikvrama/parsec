"""Pytorch Model of the Rearrangement Baseline by Abdo et al. 2016."""

from typing import Literal
from math import sqrt
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class CFModel(nn.Module):
    """Pytorch Implementation of the Collaborative Filtering approach by Abdo et al. 2016.

    Publication Source: https://journals.sagepub.com/doi/10.1177/0278364916649248.
    """

    def __init__(
        self,
        hidden_dimension: int,
        num_pairs: int,
        num_users: int,
        lambda_reg: float = 1e-5,
        mode: Literal["train", "test"] = "train",
        new_user : bool=False,
    ):
        """Initialize model parameters.

        Args:
            hidden_dimension: Dimension of the learned preference vector.
            num_pairs: Number of all object-object pairs in the data.
            num_users: Number of organizational schemas in the data.
            lambda_reg: Weight regularization constant. Defaults to 1e-5.
            mode: Mode of the model. Defaults to "train.
            new_user: If True, the model will have extra weights for a new user.
                Defaults to False.
        """

        super().__init__()
        requires_grad = mode == "train"  # freeze parameters for new users.

        # learnable variables
        biases_obj_pair = torch.randn(
            num_pairs, 1, dtype=torch.double, requires_grad=requires_grad, device=DEVICE
        )  # b_i
        biases_schema = torch.randn(
            1, num_users, dtype=torch.double, requires_grad=requires_grad, device=DEVICE
        )  # b_j
        obj_preference_matrix = torch.randn(
            hidden_dimension,
            num_pairs,
            dtype=torch.double,
            requires_grad=requires_grad,
            device=DEVICE,
        )  # s_i
        schema_preference_matrix = torch.randn(
            hidden_dimension,
            num_users,
            dtype=torch.double,
            requires_grad=requires_grad,
            device=DEVICE,
        )  # t_j

        # model dimensions
        self.num_pairs = num_pairs
        self.num_users = num_users
        self.hidden_dimension = hidden_dimension
        self.lambda_reg = lambda_reg

        # make torch parameters
        self.biases_obj_pair = nn.Parameter(
            biases_obj_pair, requires_grad=requires_grad
        )
        self.biases_schema = nn.Parameter(
            biases_schema, requires_grad=requires_grad
        )
        self.obj_preference_matrix = nn.Parameter(
            obj_preference_matrix, requires_grad=requires_grad
        )
        self.schema_preference_matrix = nn.Parameter(
            schema_preference_matrix, requires_grad=requires_grad
        )

        # random initialization
        nn.init.kaiming_uniform_(self.biases_obj_pair, a=sqrt(5))
        nn.init.kaiming_uniform_(self.biases_schema, a=sqrt(5))
        nn.init.kaiming_uniform_(self.obj_preference_matrix, a=sqrt(5))
        nn.init.kaiming_uniform_(self.schema_preference_matrix, a=sqrt(5))

        if new_user:
            self._add_new_user()
        self.new_user = new_user

    def _add_new_user(self):
        """Create new parameters for training 1 new user preference."""
        self.is_new_user = True
        bias_schema_unew = torch.randn(
            1, 1, dtype=torch.double, requires_grad=True, device=DEVICE
        )
        schema_preference_matrix_unew = torch.randn(
            self.hidden_dimension,
            1,
            dtype=torch.double,
            requires_grad=True,
            device=DEVICE,
        )
        self.bias_schema_unew = nn.Parameter(
            bias_schema_unew, requires_grad=True
        )
        self.schema_preference_matrix_unew = nn.Parameter(
            schema_preference_matrix_unew, requires_grad=True
        )
        nn.init.kaiming_uniform_(self.bias_schema_unew, a=sqrt(5))
        nn.init.kaiming_uniform_(self.schema_preference_matrix_unew, a=sqrt(5))

    def forward(self):
        """Returns predicted ratings for all object-object pairs per schema."""

        r_pred = (
            self.biases_obj_pair.repeat(1, self.num_users)
            + self.biases_schema.repeat(self.num_pairs, 1)
            + torch.matmul(self.obj_preference_matrix.T, self.schema_preference_matrix)
        )
        if r_pred.size() != (
            self.num_pairs,
            self.num_users,
        ):
            raise ValueError(
                f"Expected shape ({self.num_pairs}, {self.num_users}), got {r_pred.size()}"
            )

        if self.new_user:
            r_pred = torch.concat(
                [
                    r_pred,
                    (
                        self.biases_obj_pair
                        + self.bias_schema_unew
                        + torch.matmul(
                            self.obj_preference_matrix.T,
                            self.schema_preference_matrix_unew,
                        )
                    ),
                ],
                dim=1,
            )  # TODO: do we need to mention grad True here?
            if r_pred.size() != (
                self.num_pairs,
                self.num_users + 1,
            ):
                raise ValueError(
                    f"Expected shape ({self.num_pairs}, {self.num_users + 1}), got {r_pred.size()}"
                )
        return r_pred

    def calculate_loss(self, r_pred, r_actual, non_neg_indices):
        """Calculates MSE loss for predictions of known rating values.

        Rating matrices are of the shape (num_pairs, num_users).
        Args:
            r_pred: Predicted ratings from the forward function.
            r_actual: Actual ratings for all object-object pairs per schema.
            nonneg_indices_ravel: Indices of known ratings in flattened r_pred.
            nonneg_indices_xy: Row and Col indices of known ratings in r_pred.
        """
        assert r_pred.shape == r_actual.shape, f"Shapes of r_pred ({r_pred.shape}) and r_actual ({r_actual.shape}) do not match."
        # MSE Loss.
        non_neg_mask = torch.zeros_like(r_actual, requires_grad=False, device=DEVICE)
        non_neg_mask[non_neg_indices] = 1
        mse_loss = torch.sum(
            nn.MSELoss(reduction="none")(r_pred, r_actual) * non_neg_mask
        )
        # Regularization Loss.
        # TODO: Check if this is the correct regularization.
        regularization_loss = (
            torch.sum(self.biases_obj_pair**2)
            + torch.sum(self.biases_schema**2)
            + torch.sum(self.obj_preference_matrix**2)
            + torch.sum(self.schema_preference_matrix**2)
        )
        return mse_loss, mse_loss + 0.5 * self.lambda_reg * regularization_loss


def add_user_to_model(
    model: CFModel,
    new_ratings: np.ndarray,
    non_neg_indices: np.ndarray,
    learning_rate: float = 0.01,
    convergence_threshold: float = 1e-3,
):
    """Add a new user to the model."""
    # Create a new model with the new user.
    new_model = CFModel(
        hidden_dimension=model.hidden_dimension,
        num_pairs=model.num_pairs,
        num_users=model.num_users,
        lambda_reg=model.lambda_reg,
        mode="test",
        new_user=True,
    ).double()
    new_model.load_state_dict(model.state_dict(), strict=False)

    new_ratings = torch.tensor(
        new_ratings, dtype=torch.double, device=DEVICE, requires_grad=False
    )
    assert (
        new_ratings.size()[1] == 1
    ), f"New ratings should be a row vector (X, 1), received {new_ratings.size()}."
    optimizer = torch.optim.LBFGS(new_model.parameters(), lr=learning_rate)

    def _closure():
        optimizer.zero_grad()
        ratings_pred = new_model.forward()
        _, total_loss = new_model.calculate_loss(
            ratings_pred[:, -1].unsqueeze(1), new_ratings, non_neg_indices
        )  # Calculate loss for the new user (i.e. last column of matrix).
        total_loss.backward()
        return total_loss

    ep = 1
    loss_arr_train = []
    while True:
        loss = optimizer.step(_closure)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN, failed to add user to model.")
        loss_arr_train.append(loss.item())
        if len(loss_arr_train) > 1:
            if (
                abs(loss_arr_train[-1] - loss_arr_train[-2]) / loss_arr_train[-2]
                < convergence_threshold
            ):
                break
        ep += 1
    return new_model
