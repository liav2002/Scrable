import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Model.Models.base_model import BaseModel


class RegressionModel(BaseModel):
    """
    Linear Regression model implementation.
    """

    def __init__(self):
        """
        Initialize the Regression model.
        """
        self.model = LinearRegression()

    def fit(self, X, y) -> None:
        """
        Fit the Linear Regression model to the training data.
        """
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained Linear Regression model.
        """
        return self.model.predict(X)

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the Linear Regression model on the given data.
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
