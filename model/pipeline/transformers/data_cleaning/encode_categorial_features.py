from model.pipeline.transformers.abstract_tranformer import AbstractTransformer
from typing import List, Dict


class EncodeCategoricalFeatures(AbstractTransformer):
    """
    Transformer to encode categorical features from `games_df` and merge
    the encoded features into `train_df`.
    """

    def __init__(self, columns: List[str]):
        """
        Parameters:
        columns (list of str): List of column names in `games_df` to encode.
        """
        self.columns = columns
        self.encoders = {}

    def fit(self, data: Dict[str, object], y: None = None) -> "EncodeCategoricalFeatures":
        """
        Fit method to initialize encoders based on `games_df`.

        Parameters:
        data (dict): A dictionary containing `games_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        games_df = data["games_df"]
        for col in self.columns:
            if col in games_df.columns:
                self.encoders[col] = games_df[col].astype("category").cat.codes
        return self

    def transform(self, data: Dict[str, object]) -> Dict[str, object]:
        """
        Encode specified categorical features in `games_df` and add them to `train_df`.

        Parameters:
        data (dict): A dictionary containing `train_df` and `games_df`.

        Returns:
        dict: Updated data dictionary with encoded features added to `train_df`.
        """
        train_df = data["train_df"].copy()
        games_df = data["games_df"].copy()

        for col in self.columns:
            if col in games_df.columns:
                games_df[f"{col}_encoded"] = games_df[col].astype("category").cat.codes
                train_df = train_df.merge(
                    games_df[["game_id", f"{col}_encoded"]],
                    on="game_id",
                    how="left"
                )

        data["train_df"] = train_df
        data["games_df"] = games_df
        return data
