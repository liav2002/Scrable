from typing import Dict
import pandas as pd

from src.model.pipeline.transformers.abstract_tranformer import AbstractTransformer


class AddMeanRackUsage(AbstractTransformer):
    """
    Transformer to add `rack_len`, `rack_usage` to `turns_df`,
    and `mean_rack_usage` to `train_df`.
    """

    def __init__(self, bots_and_scores: Dict[str, int]):
        """
        Parameters:
        bots_and_scores (dict): Dictionary mapping bot names to scores.
        """
        self.bots_and_scores = bots_and_scores

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "AddMeanRackUsage":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): A dictionary containing `train_df` and `turns_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add `rack_len`, `rack_usage` to `turns_df`,
        and `mean_rack_usage` to `train_df`.

        Parameters:
        data (dict): A dictionary containing `train_df` and `turns_df`.

        Returns:
        dict: The updated dictionary with modified `train_df` and `turns_df`.
        """
        turns_df = data["turns_df"].copy()
        train_df = data["train_df"].copy()

        turns_df["rack_len"] = turns_df["rack"].str.len()
        turns_df["rack_usage"] = turns_df["move"].str.len() / turns_df["rack_len"]
        turns_df["rack_usage"] = turns_df["rack_usage"].fillna(0)

        mean_rack_usage = (
            turns_df.groupby(['game_id', 'nickname'])['rack_usage']
            .mean()
            .reset_index()
        )

        user_mean_rack_usage = mean_rack_usage[~mean_rack_usage["nickname"].isin(self.bots_and_scores.keys())]
        user_mean_rack_usage = user_mean_rack_usage.rename(columns={
            'rack_usage': 'user_mean_rack_usage',
            'nickname': 'user_name'
        })

        train_df = train_df.merge(user_mean_rack_usage, on=['game_id', 'user_name'], how='left')

        data["turns_df"] = turns_df
        data["train_df"] = train_df
        return data
