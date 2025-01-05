from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class AbstractTransformer(BaseEstimator, TransformerMixin, ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass
