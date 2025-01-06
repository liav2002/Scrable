import pandas as pd
from sklearn.model_selection import train_test_split

def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_handler,
):
    """
    Run and evaluate models.

    Args:
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed test data.
        model_handler: Model class instance to use for training and evaluation.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): DataFrame summarizing model results.
            - best_model_name (str): Name of the best model.
            - best_rmse (float): RMSE score of the best model.
            - test_predictions (list): Predictions on the test dataset.
    """
    x = train_df.drop(columns=["user_rating"])
    y = train_df["user_rating"]

    # Train-test split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    print("Training and evaluating model...")
    model_handler.fit(x_train, y_train)
    metrics = model_handler.evaluate(x_val, y_val)

    # Collect results
    results_df = pd.DataFrame([{"Model": "RegressionModel", **metrics}])
    best_model_name = "RegressionModel"
    best_rmse = metrics["RMSE"]

    # Train on full data and predict on test set
    model_handler.fit(x, y)
    x_test = test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = model_handler.predict(x_test)

    return results_df, best_model_name, best_rmse, test_predictions
