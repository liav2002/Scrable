import optuna
import pandas as pd
from typing import Dict
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from src.model.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation.
    """

    def __init__(self, **params):
        """
        Initialize the XGBoost model with given parameters.

        Args:
            **params (dict): Parameters for the XGBRegressor model.
        """
        self.params = params
        self.best_params = None
        self.model = XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the XGBoost model to the training data.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target variable for training.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for the given feature matrix.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.Series: Predicted target values.
        """
        return pd.Series(self.model.predict(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int, scoring: dict, return_train_score: bool) -> dict:
        """
        Evaluate the XGBoost model using cross-validation.

        Args:
            X (pd.DataFrame): Feature matrix for evaluation.
            y (pd.Series): True target values.
            cv_folds (int): Number of cross-validation folds.
            scoring (dict): Scoring metrics for evaluation.
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

    def search_best_params(self, X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
        """
        Perform hyperparameter tuning for the XGBoost model using Optuna.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target variable for training.
            config (dict): Hyperparameter tuning configuration.

        Returns:
            dict: Best parameters found via hyperparameter tuning.
        """
        # Parse relevant configurations for hyperparameter tuning
        search_space = config["hyperparameter_tuning"]["xgboost_search_space"]
        learning_rate_range = list(map(float, search_space["learning_rate_range"]))
        max_depth_range = search_space["max_depth"]
        n_estimators_range = search_space["n_estimators_range"]
        subsample_range = search_space["subsample_range"]
        colsample_bytree_range = search_space["colsample_bytree_range"]

        def objective(trial):
            # Suggest hyperparameters from the search space
            max_depth = trial.suggest_int("max_depth", *max_depth_range)
            learning_rate = trial.suggest_float("learning_rate", *learning_rate_range, log=True)
            n_estimators = trial.suggest_int("n_estimators", *n_estimators_range)
            subsample = trial.suggest_float("subsample", *subsample_range)
            colsample_bytree = trial.suggest_float("colsample_bytree", *colsample_bytree_range)

            # Build the model with suggested hyperparameters
            model = XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
            )

            # Perform cross-validation and evaluate
            scores = cross_validate(
                model,
                X,
                y,
                cv=config["cross_validation"]["cv_folds"],
                scoring=config["cross_validation"]["scoring"]["rmse"],
                return_train_score=config["cross_validation"]["return_train_score"]
            )

            # Return the mean of the test scores (neg RMSE)
            return -scores["test_score"].mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config["hyperparameter_tuning"]["trials"])
        self.best_params = study.best_params

        # Update the model with the best parameters
        self.model.set_params(**self.best_params)
        return self.best_params

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
