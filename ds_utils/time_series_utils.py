from typing import Union, List

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTSProcessor(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, x: pd.DataFrame, y=None, *args, **kwargs) -> pd.DataFrame:
        return self.fit(x).transform(x)


def calculate_pearson_correlation(input_df: pd.DataFrame, column_one: str, column_two: str, 
                                  window: int = None, *args, **kwargs) -> Union[float, np.ndarray]:
    if window:
        return input_df[column_one].rolling(window=window, *args, **kwargs).corr(input_df[column_two])
    return input_df[[column_one, column_two]].corr()[column_two].iloc[0]


def search_for_optimal_lag(input_df: pd.DataFrame, column_one: str, column_two: str, max_abs_shift: int = 30) -> int:
    best_shift = 0
    best_corr = 0
    for s in range(-max_abs_shift, max_abs_shift, 1):
        shift_df = input_df[[column_one, column_two]].copy()
        shift_df.loc[:, column_two] = shift_df[column_two].shift(s)
        pearson_corr = calculate_pearson_correlation(shift_df, column_one, column_two)
        if abs(pearson_corr) > best_corr:
            best_corr = abs(pearson_corr)
            best_shift = s
    return best_shift


def winsorize(input_df: pd.DataFrame, columns: Union[str, List[str]], quantile: float = 0.99) -> pd.DataFrame:
    input_df[columns] = input_df[columns].clip(upper=input_df[columns].quantile(quantile).tolist(), axis=1)
    return input_df
