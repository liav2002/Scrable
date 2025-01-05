from Model.Pipeline.Transformers.abstract_tranformer import AbstractTransformer
from typing import List, Dict


class DropFeaturesFromTrain(AbstractTransformer):
    """
    Transformer to drop specified columns from the `train_df`.
    """

    def __init__(self, columns_to_drop: List[str]):
        """
        Parameters:
        columns_to_drop (list of str): List of column names to drop from `train_df`.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X: Dict, y: None = None) -> "DropFeaturesFromTrain":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): Data dictionary containing `train_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict) -> Dict:
        """
        Drop specified columns from `train_df`.

        Parameters:
        data (dict): A dictionary containing `train_df`.

        Returns:
        dict: Updated data dictionary with columns dropped from `train_df`.
        """
        train_df = data["train_df"].copy()
        columns_to_drop = [col for col in self.columns_to_drop if col in train_df.columns]
        train_df = train_df.drop(columns=columns_to_drop)
        data["train_df"] = train_df
        return data
