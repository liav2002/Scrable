from model.pipeline.transformers.abstract_tranformer import AbstractTransformer
import pandas as pd
from typing import Dict


class SplitAndTransformTrain(AbstractTransformer):
    """
    Transformer to split the train dataset into bot and user data,
    rename columns, and merge them back into a unified structure.
    """

    def __init__(self, bots_and_scores: Dict[str, int]):
        """
        Parameters:
        bots_and_scores (dict): Dictionary mapping bot names to scores.
        """
        self.bots_and_scores = bots_and_scores

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "SplitAndTransformTrain":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): A dictionary containing `train_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Split and transform the train dataset.

        Parameters:
        data (dict): A dictionary containing `train_df`.

        Returns:
        dict: Updated data dictionary with transformed `train_df`.
        """
        train_df = data["train_df"].copy()

        user_df = train_df[~train_df["nickname"].isin(self.bots_and_scores.keys())]
        user_df = user_df.rename(
            columns={"nickname": "user_name", "score": "user_score", "rating": "user_rating"}
        )

        bot_df = train_df[train_df["nickname"].isin(self.bots_and_scores.keys())]
        bot_df = bot_df.rename(
            columns={"nickname": "bot_name", "score": "bot_score", "rating": "bot_rating"}
        )

        transformed_train_df = pd.merge(bot_df, user_df, on="game_id", how="inner")

        data["train_df"] = transformed_train_df
        return data
