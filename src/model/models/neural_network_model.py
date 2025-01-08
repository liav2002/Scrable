import optuna
import pandas as pd
from typing import Dict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from src.model.models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural Network model implementation.
    """

    def __init__(self, **params):
        """
        Initialize the Neural Network model with parameters for MLPRegressor.
        """
        self.params = params
        self.best_params = None
        self.model = MLPRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the Neural Network model to the training data.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained Neural Network model.
        """
        return pd.Series(self.model.predict(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int, scoring: Dict[str, str],
                 return_train_score: bool) -> Dict[str, float]:
        """
        Evaluate the Neural Network model using cross-validation.

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

    def search_best_params(self, X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
        """
        Perform hyperparameter tuning for the Neural Network model using Optuna.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target variable for training.
            config (dict): Hyperparameter tuning configuration.

        Returns:
            dict: Best parameters found via hyperparameter tuning.
        """
        # Parse relevant configurations for hyperparameter tuning
        search_space = config["hyperparameter_tuning"]["neural_network_search_space"]
        hidden_layer_sizes_options = search_space["hidden_layer_sizes"]
        activation_options = search_space["activation"]
        learning_rate_init_range = list(map(float, search_space["learning_rate_init_range"]))

        def objective(trial):
            # Suggest hyperparameters from the search space
            hidden_layer_sizes = tuple(
                trial.suggest_categorical("hidden_layer_sizes", hidden_layer_sizes_options)
            )
            activation = trial.suggest_categorical("activation", activation_options)
            learning_rate_init = trial.suggest_float(
                "learning_rate_init", *learning_rate_init_range, log=True
            )

            # Build the model with suggested hyperparameters
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=config["models"]["Neural Network"]["params"]["solver"],
                learning_rate_init=learning_rate_init,
                max_iter=config["models"]["Neural Network"]["params"]["max_iter"],
                random_state=config["models"]["Neural Network"]["params"]["random_state"],
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
            dict: Parameters of the Neural Network model.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> None:
        """
        Set the parameters of this estimator.
        """
        self.model.set_params(**params)
