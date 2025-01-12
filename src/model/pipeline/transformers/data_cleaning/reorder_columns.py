from typing import Dict
import pandas as pd

from src.model.pipeline.transformers.abstract_tranformer import AbstractTransformer


class ReorderColumns(AbstractTransformer):
    """
    Transformer to reorder columns in the dataset so that the `target_column`
    column is always the last column.
    """

    def __init__(self, target_column: str = "user_rating"):
        """
        Parameters:
        target_column (str): The column to push to the last position.
        """
        self.target_column = target_column

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "ReorderColumns":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): A dictionary containing a `train_df` DataFrame.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Reorder columns in the `train_df` so that the target column is last.

        Parameters:
        data (dict): A dictionary containing a `train_df` DataFrame.

        Returns:
        dict: Updated data dictionary with reordered `train_df`.
        """
        train_df = data["train_df"]

        if self.target_column in train_df.columns:
            columns = [col for col in train_df.columns if col != self.target_column]
            data["train_df"] = train_df[columns + [self.target_column]]

        return data
