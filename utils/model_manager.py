import importlib
import pandas as pd
from utils.logger import Logger

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
        models: dict,
        config: dict,
        logger: Logger
) -> pd.DataFrame:
    """
    Train and evaluate multiple models using their respective evaluate methods,
    and determine the best model by a configurable metric.

    Args:
        logger: (Logger): logger object.
        train_df (pd.DataFrame): Processed training data.
        models (dict): A dictionary of models to evaluate.
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame summarizing model results.
    """
    # Load cross-validation parameters from config
    cv_config = config["cross_validation"]
    cv_folds = cv_config["cv_folds"]
    scoring = cv_config["scoring"]
    return_train_score = cv_config["return_train_score"]
    best_metric_key = cv_config["best_metric_key"]

    logger.log("Preparing training data...")
    x = train_df.drop(columns=["user_rating"])
    y = train_df["user_rating"]

    results = []
    for model_name, model_handler in models.items():
        logger.log(f"Evaluating {model_name} with {cv_folds} folds...")
        metrics = model_handler.evaluate(
            x, y, cv_folds=cv_folds, scoring=scoring, return_train_score=return_train_score
        )
        logger.log(f"Model {model_name}: Metrics = {metrics}")
        results.append({"Model": model_name, **metrics})

    results_df = pd.DataFrame(results)

    # Find the best model using the configurable metric
    if best_metric_key not in scoring:
        logger.log(f"Error: Best metric key '{best_metric_key}' is not in scoring metrics.")
        raise ValueError(f"Best metric key '{best_metric_key}' is not in scoring metrics.")

    best_model_row = results_df.loc[results_df[best_metric_key].idxmin()]
    best_model_name = best_model_row["Model"]

    # Save the best model name to an output file
    with open("output/best_model.txt", "w") as file:
        file.write(best_model_name)

    logger.log(f"Best model is {best_model_name} with {best_metric_key.upper()}: {best_model_row[best_metric_key]:.4f}")

    return results_df


def train_and_predict(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        models: dict,
        logger: Logger,
) -> list:
    """
    Train the best model on the entire training data and predict on the test data.

    Args:
        logger (Logger): logger object.
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed test data.
        models (dict): A dictionary of models to evaluate.

    Returns:
        list: Predictions on the test dataset by the best model.
    """
    # Load the best model name from the output file
    try:
        with open("output/best_model.txt", "r") as file:
            best_model_name = file.read().strip()
    except FileNotFoundError:
        logger.log("Error: Best model file not found. Run 'run_models' first.")
        raise ValueError("Best model file not found. Run 'run_models' first.")

    # Get the best model
    best_model_handler = models.get(best_model_name)
    if not best_model_handler:
        logger.log(f"Error: Best model '{best_model_name}' not found in models dictionary.")
        raise ValueError(f"Best model '{best_model_name}' not found in models dictionary.")

    logger.log(f"Training the best model '{best_model_name}' on the full training data...\n")
    x = train_df.drop(columns=["user_rating"])
    y = train_df["user_rating"]
    best_model_handler.fit(x, y)

    # Predict on the test data
    logger.log("Making predictions on the test data...")
    x_test = test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = best_model_handler.predict(x_test)

    return test_predictions
