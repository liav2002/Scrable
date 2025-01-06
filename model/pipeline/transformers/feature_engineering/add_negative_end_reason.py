from model.pipeline.transformers.abstract_tranformer import AbstractTransformer
from typing import Dict
import pandas as pd


class AddNegativeEndReason(AbstractTransformer):
    """
    Transformer to create the `negative_end_reason` feature from `games_df`
    and merge it into `train_df`.
    """

    def __init__(self):
        """
        Initialize the transformer.
        """
        pass

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "AddNegativeEndReason":
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
        Add the `negative_end_reason` feature from `games_df` to `train_df`.

        Parameters:
        data (dict): A dictionary containing `train_df` and `games_df`.

        Returns:
        dict: Updated data dictionary with `negative_end_reason` added to `train_df`.
        """
        train_df = data["train_df"].copy()
        games_df = data["games_df"].copy()

        games_df["negative_end_reason"] = games_df["game_end_reason"].apply(
            lambda x: 1 if x in ["RESIGNED", "TIME", "CONSECUTIVE_ZEROS"] else 0
        )

        train_df = train_df.merge(
            games_df[["game_id", "negative_end_reason"]],
            on="game_id",
            how="left"
        )

        data["train_df"] = train_df
        data["games_df"] = games_df
        return data
