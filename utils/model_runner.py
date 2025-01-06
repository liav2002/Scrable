import importlib
import pandas as pd
from sklearn.model_selection import train_test_split


def get_model_instance(model_class_path: str):
    """
    Dynamically load a model class and return its instance.

    Args:
        model_class_path (str): The full path of the model class (e.g., "module.submodule.ClassName").

    Returns:
        object: An instance of the specified model class.
    """
    module_name, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def run_models(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        models: dict,
) -> tuple:
    """
    Train and evaluate multiple models, and determine the best model by RMSE.

    Args:
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed test data.
        models (dict): A dictionary of models to evaluate.
                       Keys are model names, and values are model instances.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): DataFrame summarizing model results.
            - best_model_name (str): Name of the best model.
            - best_rmse (float): RMSE score of the best model.
            - test_predictions (list): Predictions on the test dataset by the best model.
    """
    # Prepare training and validation data
    x = train_df.drop(columns=["user_rating"])
    y = train_df["user_rating"]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Evaluate all models
    results = []
    for model_name, model_handler in models.items():
        print(f"Training and evaluating {model_name}...")
        model_handler.fit(x_train, y_train)
        metrics = model_handler.evaluate(x_val, y_val)
        results.append({"Model": model_name, **metrics})

    # Create a summary DataFrame
    results_df = pd.DataFrame(results)

    # Find the best model by RMSE
    best_model_row = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_model_row["Model"]
    best_rmse = best_model_row["RMSE"]

    # Train the best model on the full training data
    best_model_handler = models[best_model_name]
    best_model_handler.fit(x, y)

    # Predict on the test data
    x_test = test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = best_model_handler.predict(x_test)

    return results_df, best_model_name, best_rmse, test_predictions
