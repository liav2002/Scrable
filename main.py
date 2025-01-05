import pandas as pd
from sklearn.model_selection import train_test_split
from Model.model import ModelHandler
from Model.pipline import DataPipeline

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

    # Create pipline
    pipeline = DataPipeline(bots_and_scores, turns_df, games_df)

    # Process train and test datasets
    processed_train_df = pipeline.process_train_data(train_df)
    processed_test_df = pipeline.process_test_data(test_df)

    # Prepare training data
    X = processed_train_df.drop(columns=["user_rating"])
    y = processed_train_df["user_rating"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model_handler = ModelHandler()
    model_handler.fit(X_train, y_train)
    metrics = model_handler.evaluate(X_val, y_val)

    print("Model Performance on Validation Set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Predict on test data
    X_test = processed_test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = model_handler.predict(X_test)

    print("\nSample Test Predictions:")
    for idx, pred in enumerate(test_predictions[:10]):
        print(f"Test Sample {idx + 1}: Predicted Rating = {pred:.2f}")


if __name__ == "__main__":
    main()
