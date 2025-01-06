import pandas as pd
from sklearn.model_selection import train_test_split
from Model.Pipeline.pipeline import DataPipeline
from Model.Models.regression_model import RegressionModel


def load_data():
    """
    Load source data from the data directory.
    """
    train_df = pd.read_csv("data/source_data/train.csv")
    test_df = pd.read_csv("data/source_data/test.csv")
    games_df = pd.read_csv("data/source_data/games.csv")
    turns_df = pd.read_csv("data/source_data/turns.csv")
    return train_df, test_df, games_df, turns_df


def main():
    bots_and_scores = {"BetterBot": 268240632, "STEEBot": 276067680, "HastyBot": 588506671}
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
