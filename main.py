import pyyaml
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from Model.Pipeline.pipeline import DataPipeline
from Model.Models.regression_model import RegressionModel


def load_config(file_path: str = "Config/config.yaml") -> dict:
    """
    Load the configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source data from the data directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        A tuple containing:
            train_df: DataFrame for training data.
            test_df: DataFrame for test data.
            games_df: DataFrame for games data.
            turns_df: DataFrame for turns data.
     """
    with open("Config/file_paths.yaml", "r") as file:
        paths = yaml.safe_load(file)["data_paths"]

    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])
    games_df = pd.read_csv(paths["games"])
    turns_df = pd.read_csv(paths["turns"])

    return train_df, test_df, games_df, turns_df


def main():
    config = load_config()
    bots_and_scores = config["bots_and_scores"]
    train_df, test_df, games_df, turns_df = load_data()

    # Create pipeline
    pipeline = DataPipeline(bots_and_scores, turns_df, games_df)

    # Process train and test datasets
    processed_train_df = pipeline.process_train_data(train_df)
    processed_test_df = pipeline.process_test_data(test_df)

    # Prepare training data
    x = processed_train_df.drop(columns=["user_rating"])
    y = processed_train_df["user_rating"]

    # Train-test split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Dictionary of models to evaluate
    models = {
        "Linear Regression": RegressionModel()
    }

    # Evaluate all models
    results = []
    for model_name, model_handler in models.items():
        print(f"Training and evaluating {model_name}...")
        model_handler.fit(x_train, y_train)
        metrics = model_handler.evaluate(x_val, y_val)
        results.append({
            "Model": model_name,
            **metrics
        })

    # Create a summary DataFrame for easier comparison
    results_df = pd.DataFrame(results)

    # Find the best model based on RMSE
    best_model_row = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_model_row["Model"]
    best_rmse = best_model_row["RMSE"]

    # Print the summary
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.4f})")

    # Train the best model on the entire training set and predict on test data
    best_model_handler = models[best_model_name]
    best_model_handler.fit(x, y)

    x_test = processed_test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = best_model_handler.predict(x_test)

    # Print sample predictions
    print("\nSample Test Predictions:")
    for idx, pred in enumerate(test_predictions[:10]):
        print(f"Test Sample {idx + 1}: Predicted Rating = {pred:.2f}")


if __name__ == "__main__":
    main()
