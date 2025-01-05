import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    """

    @abstractmethod
    def fit(self, X, y) -> None:
        """
        Fit the model to the training data.

        Parameters:
            X (DataFrame): Features for training.
            y (Series): Target variable.
        """
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Make predictions on the input data.

        Parameters:
            X (DataFrame): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Evaluate the model on the given data.

        Parameters:
            X (DataFrame): Features for evaluation.
            y (Series): True target values.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        pass
