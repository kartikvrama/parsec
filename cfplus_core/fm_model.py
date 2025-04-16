"""Factorization Machine model for regression."""

import pywFM
import numpy as np
from utils import constants


class FMRegressor:
    """Matrix Factorization model for regression.

    Modified from: https://github.com/jilljenn/ktm/blob/master/fm.py#L25.
    """

    def __init__(
        self,
        hidden_dimension: int = 30,
        num_iter: int = 500,
        init_stdev: float = 0.1,
        learn_rate: float = 0.03,
        seed: int = constants.SEED,
    ):
        super().__init__()
        np.random.seed(seed)
        self.hidden_dimension = hidden_dimension
        self.num_iter = num_iter
        self.init_stdev = init_stdev
        self.learn_rate = learn_rate

        # TODO: test this when loading parameters works.
        self.mu = None
        self.W = None
        self.V = None
        self.V2 = None

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        shuffle: bool = True,
    ):
        """Fit the model on the training data and simultaneously predict.

        Validation split is 5% of the training data by default.

        Args:
            X_train: Training data.
            y_train: Training labels.
            X_test: Test data.
            y_test: Test labels.
            shuffle: Whether to shuffle the data before fitting.
        """
        fm = pywFM.FM(
            task="regression",
            num_iter=self.num_iter,
            init_stdev=self.init_stdev,
            k0=True,
            k1=True,
            k2=self.hidden_dimension,
            learning_method="sgda",
            learn_rate=self.learn_rate,
            rlog=False,
            verbose=False,
            silent=True,
        )
        if shuffle:
            shuffle_indices = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
        train_len = int(0.95 * len(X_train))
        model = fm.run(
            x_train=X_train[:train_len],
            y_train=y_train[:train_len],
            x_test=X_test,
            y_test=np.ones(len(X_test)),  # Dummy values.
            x_validation_set=X_train[train_len:],  # Validation set
            y_validation_set=y_train[train_len:],
        )
        return model.predictions

    def predict(self, X):
        # TODO: Test this when you figure out how to fix bias=None issue.
        X_2 = X.copy()
        X_2 **= 2
        y_pred = (
            self.mu
            + X @ self.W
            + 0.5*(
                np.power(X @ self.V, 2).sum(axis=1)
                - (X_2 @ self.V2).sum(axis=1)
            ).A1
        )
        return y_pred
