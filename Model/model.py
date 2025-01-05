from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class ModelHandler:
    """
    Class to handle model training, validation, and prediction.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Train the model with the given data.

        Parameters:
            X (DataFrame): Features for training.
            y (Series): Target variable.
        """
        self.model.fit(X, y)

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on the validation set.

        Parameters:
            X_val (DataFrame): Validation features.
            y_val (Series): Validation target variable.

        Returns:
            dict: Evaluation metrics.
        """
        y_val_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def predict(self, X_test):
        """
        Make predictions on the test dataset.

        Parameters:
            X_test (DataFrame): Test features.

        Returns:
            np.array: Predicted values.
        """
        return self.model.predict(X_test)
