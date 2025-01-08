import pandas as pd

from src.utils.logger import logger


def run_models_and_get_best(
        train_df: pd.DataFrame,
        models: dict,
        config: dict
) -> tuple:
    """
    Train and evaluate multiple models using their respective evaluate methods,
    and determine the best model by a configurable metric.

    Args:
        train_df (pd.DataFrame): Processed training data.
        models (dict): A dictionary of ModelHandler instances to evaluate.
        config (dict): Configuration dictionary.

    Returns:
        tuple:
            pd.DataFrame: DataFrame summarizing model results.
            ModelHandler: The best model handler instance.
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

        # Call the evaluate method of the ModelHandler
        metrics = model_handler.evaluate(
            x, y, cv_folds=cv_folds, scoring=scoring, return_train_score=return_train_score
        )

        logger.log(f"Model {model_name}: Metrics = {metrics}")
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results)

    # Validate the best metric key exists in the scoring metrics
    if best_metric_key not in scoring:
        logger.log(f"Error: Best metric key '{best_metric_key}' is not in scoring metrics.")
        raise ValueError(f"Best metric key '{best_metric_key}' is not in scoring metrics.")

    # Identify the best model using the best metric key
    best_model_row = results_df.loc[results_df[best_metric_key].idxmin()]
    best_model_name = best_model_row["model"]
    best_model_handler = models[best_model_name]

    logger.log(f"Best model is {best_model_name} with {best_metric_key.upper()}: {best_model_row[best_metric_key]:.4f}")

    return results_df, best_model_handler


def train_and_predict_with_best(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        best_model_handler
) -> list:
    """
    Train the best model on the entire training data and predict on the test data.

    Args:
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed test data.
        best_model_handler: The best model instance.

    Returns:
        list: Predictions on the test dataset by the best model.
    """
    logger.log("Training the best model on the full training data...\n")
    x = train_df.drop(columns=["user_rating"])
    y = train_df["user_rating"]
    best_model_handler.fit(x, y)

    logger.log("Making predictions on the test data...")
    x_test = test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = best_model_handler.predict(x_test)

    return test_predictions
