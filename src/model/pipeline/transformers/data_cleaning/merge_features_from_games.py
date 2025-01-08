from src.model.pipeline.transformers.abstract_tranformer import AbstractTransformer
from typing import List, Dict


class MergeFeaturesFromGames(AbstractTransformer):
    """
    Transformer to merge specified columns from `games_df` into `train_df`.
    """

    def __init__(self, columns_to_merge: List[str]):
        """
        Parameters:
        columns_to_merge (list of str): List of column names from `games_df` to merge into `train_df`.
        """
        self.columns_to_merge = columns_to_merge

    def fit(self, X: Dict[str, object], y: None = None) -> "MergeFeaturesFromGames":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): A dictionary containing `train_df` and `games_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, object]) -> Dict[str, object]:
        """
        Merge specified columns from `games_df` into `train_df`.

        Parameters:
        data (dict): A dictionary containing `train_df` and `games_df`.

        Returns:
        dict: Updated data dictionary with merged columns in `train_df`.
        """
        train_df = data["train_df"].copy()
        games_df = data["games_df"].copy()

        columns_to_merge = ["game_id"] + [col for col in self.columns_to_merge if col in games_df.columns]

        train_df = train_df.merge(
            games_df[columns_to_merge],
            on="game_id",
            how="left"
        )

        data["train_df"] = train_df
        return data
