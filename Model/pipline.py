import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Model.Transformers.Data_Cleaning.reorder_columns import ReorderColumns
from Model.Transformers.Feature_Engineering.add_is_first_play import AddIsFirstPlay
from Model.Transformers.Feature_Engineering.add_bot_difficulty import AddBotDifficulty
from Model.Transformers.Feature_Engineering.add_mean_rack_usage import AddMeanRackUsage
from Model.Transformers.Data_Cleaning.drop_features_from_train import DropFeaturesFromTrain
from Model.Transformers.Data_Cleaning.split_and_trasform_train import SplitAndTransformTrain
from Model.Transformers.Data_Cleaning.merge_features_from_games import MergeFeaturesFromGames
from Model.Transformers.Feature_Engineering.add_negative_end_reason import AddNegativeEndReason
from Model.Transformers.Data_Cleaning.encode_categorial_features import EncodeCategoricalFeatures
from Model.Transformers.Feature_Engineering.add_mean_average_letter_score import AddMeanAverageLetterScore


def create_pipeline(bots_and_scores):
    """
    Creates a machine learning pipeline for the train dataset, including transformations for turns data.

    Parameters:
        bots_and_scores (dict): Dictionary mapping bot names to their scores.

    Returns:
        Pipeline: Configured machine learning pipeline.
    """
    pipeline = Pipeline([
        ('split_transform_train', SplitAndTransformTrain(bots_and_scores)),
        ('add_mean_rack_usage', AddMeanRackUsage(bots_and_scores)),
        ('add_mean_avg_letter_score', AddMeanAverageLetterScore()),
        ('add_bot_difficulty', AddBotDifficulty(bots_and_scores=bots_and_scores)),
        ('add_is_first_play', AddIsFirstPlay(bot_names=list(bots_and_scores.keys()))),
        ('add_negative_end_reason', AddNegativeEndReason()),
        ('add_encode_features_from_games',
         EncodeCategoricalFeatures(columns=["lexicon", "game_end_reason", "rating_mode"])),
        ('merge_columns_from_games',
         MergeFeaturesFromGames(columns_to_merge=["initial_time_seconds", "game_duration_seconds", "winner"])),
        ('drop_features_from_train', DropFeaturesFromTrain(columns_to_drop=["bot_name", "user_name"])),
        ('reorder_columns', ReorderColumns(target_column="user_rating"))
    ])
    return pipeline


def load_data():
    """
    Load source data from the data directory.
    """
    train_df = pd.read_csv("../data/source_data/train.csv")
    test_df = pd.read_csv("../data/source_data/test.csv")
    games_df = pd.read_csv("../data/source_data/games.csv")
    turns_df = pd.read_csv("../data/source_data/turns.csv")
    return train_df, test_df, games_df, turns_df


def main():
    """
    Main function to execute the pipeline and create the processed train dataset.
    """
    bots_and_scores = {"BetterBot": 268240632, "STEEBot": 276067680, "HastyBot": 588506671}

    train_df, test_df, games_df, turns_df = load_data()

    pipeline = create_pipeline(bots_and_scores)

    # Process train dataset using the pipeline
    data = {
        "train_df": train_df,
        "turns_df": turns_df,
        "games_df": games_df
    }
    processed_data = pipeline.fit_transform(data)
    processed_train_df = processed_data["train_df"]

    # Process test dataset using the same pipeline
    test_data = {
        "train_df": test_df,  # Pass test_df as train_df for pipeline compatibility
        "turns_df": turns_df,
        "games_df": games_df
    }
    processed_test_data = pipeline.transform(test_data)
    processed_test_df = processed_test_data["train_df"]

    # Prepare training data for the model
    X = processed_train_df.drop(columns=["user_rating"])
    y = processed_train_df["user_rating"]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a basic Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    print("Model Performance on Validation Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Predict ratings for the processed test dataset
    X_test = processed_test_df.drop(columns=["user_rating"], errors="ignore")
    test_predictions = model.predict(X_test)

    # Print some test predictions
    print("\nSample Test Predictions:")
    for idx, pred in enumerate(test_predictions[:10]):
        print(f"Test Sample {idx + 1}: Predicted Rating = {pred:.2f}")


main()
