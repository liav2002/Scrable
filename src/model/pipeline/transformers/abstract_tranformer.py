from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class AbstractTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for custom transformers.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AbstractTransformer":
        """
        Fit the transformer to the data.

        Args:
            X (pd.DataFrame): Input features.
            y (Optional[pd.Series]): Target variable. Defaults to None.

        Returns:
            AbstractTransformer: The fitted transformer instance.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X (pd.DataFrame): Input features to transform.

        Returns:
            pd.DataFrame: Transformed features.
        """
        pass
