import pandas as pd
from sklearn.pipeline import Pipeline

from src.model.pipeline.transformers.data_cleaning.reorder_columns import ReorderColumns
from src.model.pipeline.transformers.feature_engineering.add_is_first_play import AddIsFirstPlay
from src.model.pipeline.transformers.feature_engineering.add_bot_difficulty import AddBotDifficulty
from src.model.pipeline.transformers.feature_engineering.add_mean_rack_usage import AddMeanRackUsage
from src.model.pipeline.transformers.data_cleaning.drop_features_from_train import DropFeaturesFromTrain
from src.model.pipeline.transformers.data_cleaning.split_and_trasform_train import SplitAndTransformTrain
from src.model.pipeline.transformers.data_cleaning.merge_features_from_games import MergeFeaturesFromGames
from src.model.pipeline.transformers.feature_engineering.add_negative_end_reason import AddNegativeEndReason
from src.model.pipeline.transformers.data_cleaning.encode_categorial_features import EncodeCategoricalFeatures
from src.model.pipeline.transformers.feature_engineering.add_mean_average_letter_score import AddMeanAverageLetterScore


class DataPipeline:
    """
    Class to create and handle the data pipeline for preprocessing and feature engineering.
    """

    def __init__(self, bots_and_scores: dict, turns_df: pd.DataFrame, games_df: pd.DataFrame):
        """
        Initialize the pipeline with static data.

        Parameters:
            bots_and_scores (dict): Dictionary mapping bot names to their scores.
            turns_df (pd.DataFrame): DataFrame containing turns information.
            games_df (pd.DataFrame): DataFrame containing game metadata.
        """
        self.bots_and_scores = bots_and_scores
        self.turns_df = turns_df
        self.games_df = games_df
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """
        Creates a machine learning pipeline for the train dataset.
        """
        return Pipeline([
            ('split_transform_train', SplitAndTransformTrain(self.bots_and_scores)),
            ('add_mean_rack_usage', AddMeanRackUsage(self.bots_and_scores)),
            ('add_mean_avg_letter_score', AddMeanAverageLetterScore()),
            ('add_bot_difficulty', AddBotDifficulty(bots_and_scores=self.bots_and_scores)),
            ('add_is_first_play', AddIsFirstPlay(bot_names=list(self.bots_and_scores.keys()))),
            ('add_negative_end_reason', AddNegativeEndReason()),
            ('add_encode_features_from_games',
             EncodeCategoricalFeatures(columns=["lexicon", "game_end_reason", "rating_mode"])),
            ('merge_columns_from_games',
             MergeFeaturesFromGames(columns_to_merge=["initial_time_seconds", "game_duration_seconds", "winner"])),
            ('drop_features_from_train', DropFeaturesFromTrain(columns_to_drop=["bot_name", "user_name"])),
            ('reorder_columns', ReorderColumns(target_column="user_rating"))
        ])

    def process_train_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the training dataset.

        Parameters:
            train_df (pd.DataFrame): The training dataset to process.

        Returns:
            pd.DataFrame: Processed training dataset.
        """
        data = {
            "train_df": train_df,
            "turns_df": self.turns_df,
            "games_df": self.games_df
        }
        processed_data = self.pipeline.fit_transform(data)
        return processed_data["train_df"]

    def process_test_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the test dataset.

        Parameters:
            test_df (pd.DataFrame): The test dataset to process.

        Returns:
            pd.DataFrame: Processed test dataset.
        """
        data = {
            "train_df": test_df,  # Passed as "train_df" for compatibility
            "turns_df": self.turns_df,
            "games_df": self.games_df
        }
        processed_data = self.pipeline.transform(data)
        return processed_data["train_df"]
