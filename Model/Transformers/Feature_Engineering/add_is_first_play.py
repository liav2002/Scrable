from Model.Transformers.abstract_tranformer import AbstractTransformer
from typing import List, Dict
import pandas as pd


class AddIsFirstPlay(AbstractTransformer):
    """
    Transformer to create the `is_first_play` feature from `games_df`
    and merge it into `train_df`.
    """

    def __init__(self, bot_names: List[str]):
        """
        Parameters:
        bot_names (list of str): List of bot names.
        """
        self.bot_names = bot_names

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "AddIsFirstPlay":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): A dictionary containing `train_df` and `games_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add the `is_first_play` feature to `train_df` based on `games_df`.

        Parameters:
        data (dict): A dictionary containing `train_df` and `games_df`.

        Returns:
        dict: Updated data dictionary with `is_first_play` added to `train_df`.
        """
        train_df = data["train_df"].copy()
        games_df = data["games_df"].copy()

        games_df["is_first_play"] = games_df["first"].apply(
            lambda x: 1 if x not in self.bot_names else 0
        )

        train_df = train_df.merge(
            games_df[["game_id", "is_first_play"]],
            on="game_id",
            how="left"
        )

        data["train_df"] = train_df
        return data
