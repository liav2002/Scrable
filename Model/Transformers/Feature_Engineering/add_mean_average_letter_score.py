from Model.Transformers.abstract_tranformer import AbstractTransformer
import pandas as pd
from typing import Dict


class AddMeanAverageLetterScore(AbstractTransformer):
    """
    Transformer to calculate and merge the Mean Average Letter Score feature
    from the `turns` dataset into the `train` dataset.
    """

    def __init__(self):
        """
        Initialize the transformer.
        """
        self.letter_points = {
            1: "AEILNORSTU",
            2: "DG",
            3: "BCMP",
            4: "FHVWY",
            5: "K",
            8: "JX",
            10: "QZ"
        }
        self.letter_to_points = {
            letter: points
            for points, letters in self.letter_points.items()
            for letter in letters
        }

    def _calculate_word_score(self, word: str) -> int:
        """
        Calculate the Scrabble score for a given word.

        Parameters:
        word (str): The word to calculate the score for.

        Returns:
        int: The total score of the word.
        """
        if isinstance(word, str):
            return sum(self.letter_to_points.get(letter.upper(), 0) for letter in word)
        return 0

    def fit(self, X: Dict[str, pd.DataFrame], y: None = None) -> "AddMeanAverageLetterScore":
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
        Add the Mean Average Letter Score feature to the `train` dataset.

        Parameters:
        data (dict): A dictionary containing `train_df` and `turns_df`.

        Returns:
        dict: Updated data dictionary with `mean_average_letter_score` added to `train_df`.
        """
        train_df = data["train_df"].copy()
        turns_df = data["turns_df"].copy()

        turns_df["letter_score"] = turns_df["move"].apply(self._calculate_word_score)
        turns_df["average_letter_score"] = turns_df["letter_score"] / turns_df["move"].str.len()

        turns_df["average_letter_score"] = turns_df["average_letter_score"].fillna(0)

        mean_avg_letter_score = (
            turns_df.groupby(["game_id", "nickname"])["average_letter_score"]
            .mean()
            .reset_index()
            .rename(columns={"average_letter_score": "mean_average_letter_score"})
        )

        train_df = train_df.merge(
            mean_avg_letter_score,
            left_on=["game_id", "user_name"],
            right_on=["game_id", "nickname"],
            how="left"
        )

        train_df = train_df.drop(columns=["nickname"])

        data["train_df"] = train_df
        data["turns_df"] = turns_df

        return data
