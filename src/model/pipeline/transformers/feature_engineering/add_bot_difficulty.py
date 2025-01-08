from src.model.pipeline.transformers.abstract_tranformer import AbstractTransformer
from typing import Dict


class AddBotDifficulty(AbstractTransformer):
    """
    Transformer to add the `bot_difficulty` feature to the train dataset
    based on bot scores.
    """

    def __init__(self, bots_and_scores: Dict[str, int]):
        """
        Parameters:
        bots_and_scores (dict): Dictionary mapping bot names to scores.
        """
        self.bots_and_scores = bots_and_scores
        self.bots_difficulty = {
            bot: self._score_to_difficulty(score)
            for bot, score in bots_and_scores.items()
        }

    def _score_to_difficulty(self, score: int) -> int:
        """
        Map bot score to difficulty levels:
        - 1: Low difficulty
        - 2: Medium difficulty
        - 3: High difficulty

        Parameters:
        score (int): The bot's score.

        Returns:
        int: Difficulty level (1, 2, or 3).
        """
        if score == max(self.bots_and_scores.values()):
            return 3  # High difficulty
        elif score == min(self.bots_and_scores.values()):
            return 1  # Low difficulty
        else:
            return 2  # Medium difficulty

    def fit(self, X: Dict[str, object], y: None = None) -> "AddBotDifficulty":
        """
        Fit method for compatibility with the pipeline.

        Parameters:
        X (dict): Data dictionary containing `train_df`.
        y: Ignored.

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, data: Dict[str, object]) -> Dict[str, object]:
        """
        Add `bot_difficulty` feature to the train dataset.

        Parameters:
        data (dict): A dictionary containing `train_df`.

        Returns:
        dict: Updated data dictionary with `bot_difficulty` added to `train_df`.
        """
        train_df = data["train_df"].copy()
        train_df["bot_difficulty"] = train_df["bot_name"].map(self.bots_difficulty)
        data["train_df"] = train_df
        return data
