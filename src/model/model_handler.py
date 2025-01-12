from typing import Dict, Union
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import cross_validate


class ModelHandler:
    """
    A generic handler for managing machine learning models, including training, evaluation,
    and hyperparameter tuning.
    """

    def __init__(self, model: type, params: Dict[str, Union[int, float, str]]) -> None:
        """
        Initializes the ModelHandler.

        Args:
            model (type): A machine learning model class (e.g., from scikit-learn, XGBoost, etc.).
            params (Dict[str, Union[int, float, str]]): Parameters for the model.
        """
        self.model = model(**params)
        self.params = params
        self.best_params = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int,
        scoring: Dict[str, str],
        return_train_score: bool
    ) -> Dict[str, float]:
        """
        Evaluates the model using cross-validation.

        Args:
            X (pd.DataFrame): Feature matrix for evaluation.
            y (pd.Series): True target values.
            cv_folds (int): Number of cross-validation folds.
            scoring (Dict[str, str]): Scoring metrics for evaluation.
            return_train_score (bool): Whether to return train scores.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        scores = cross_validate(
            self.model,
            X,
            y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=return_train_score
        )

        metrics = {key: -scores[f"test_{key}"].mean() for key in scoring.keys()}
        if return_train_score:
            metrics.update({f"Train {key}": -scores[f"train_{key}"].mean() for key in scoring.keys()})

        return metrics

    def search_best_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Dict[str, Dict[str, Union[str, list]]]],
        cv_folds: int,
        scoring: str,
        n_trials: int
    ) -> Dict[str, Union[int, float, str]]:
        """
        Performs hyperparameter tuning using Optuna.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target variable for training.
            search_space (Dict[str, Dict[str, Dict[str, Union[str, list]]]]): Hyperparameter search space.
            cv_folds (int): Number of cross-validation folds.
            scoring (str): Scoring metric for evaluation.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Union[int, float, str]]: Best parameters found via hyperparameter tuning.
        """
        model_space = search_space[self.model.__class__.__name__]

        def objective(trial):
            trial_params = {}
            for param_name, param_info in model_space.items():
                param_type = param_info["type"]
                param_value = param_info["value"]
                if param_type == "int":
                    trial_params[param_name] = trial.suggest_int(param_name, *param_value)
                elif param_type == "float":
                    trial_params[param_name] = trial.suggest_float(param_name, *param_value, log=True)
                elif param_type == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_value)

            model = self.model.__class__(**trial_params)
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv_folds,
                scoring=scoring
            )
            return -scores["test_score"].mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params

        # Update the model with the best parameters
        self.model.set_params(**self.best_params)
        return self.best_params
