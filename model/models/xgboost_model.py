import numpy as np
from typing import Dict
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from model.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model with parameters.
        """
        self.model = XGBRegressor(**kwargs)

    def fit(self, X, y) -> None:
        """
        Fit the XGBoost model to the training data.
        """
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        """
        return self.model.predict(X)

    def evaluate(self, X, y, cv_folds: int, scoring: Dict[str, str], return_train_score: bool) -> Dict[str, float]:
        """
        Evaluate the XGBoost model using cross-validation.

        Parameters:
            X (DataFrame): Features for evaluation.
            y (Series): True target values.
            cv_folds (int): Number of cross-validation folds.
            scoring (Dict[str, str]): Scoring metrics for evaluation.
            return_train_score (bool): Whether to return train scores.

        Returns:
            dict: A dictionary containing cross-validation metrics.
        """
        scores = cross_validate(
            self.model,
            X,
            y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=return_train_score,
        )

        metrics = {key: -scores[f"test_{key}"].mean() for key in scoring.keys()}
        if return_train_score:
            metrics.update({f"Train {key}": -scores[f"train_{key}"].mean() for key in scoring.keys()})

        return metrics

    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for this estimator.

        Returns:
            dict: Parameters of the XGBoost model.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> None:
        """
        Set the parameters of this estimator.
        """
        self.model.set_params(**params)
