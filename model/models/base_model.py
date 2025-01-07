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
    def evaluate(self, X, y, cv_folds: int, scoring: Dict[str, str], return_train_score: bool) -> Dict[str, float]:
        """
        Evaluate the model using cross-validation.

        Parameters:
            X (DataFrame): Features for evaluation.
            y (Series): True target values.
            cv_folds (int): Number of cross-validation folds.
            scoring (Dict[str, str]): Scoring metrics for evaluation.
            return_train_score (bool): Whether to return train scores.

        Returns:
            dict: A dictionary containing cross-validation metrics.
        """
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for the model.

        Parameters:
            deep (bool): Whether to include nested parameters.

        Returns:
            dict: A dictionary of parameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set parameters for the model.

        Parameters:
            **params: model parameters to set.
        """
        pass

    @abstractmethod
    def search_best_params(self, X: pd.DataFrame, y: pd.Series, logger) -> Dict:
        """
        Perform hyperparameter tuning using Optuna.
        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target variable for training.
            logger: Logger instance to log the tuning process.
        Returns:
            dict: Best parameters found via hyperparameter tuning.
        """
        pass
