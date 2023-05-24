import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePandasTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, x: pd.DataFrame, y=None, *args, **kwargs) -> pd.DataFrame:
        return self.fit(x, y).transform(x, y)
