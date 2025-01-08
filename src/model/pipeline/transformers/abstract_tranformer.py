from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Optional
import pandas as pd
import numpy as np


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
