from typing import Union, List

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import acf, pacf
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from .base import BasePandasTransformer


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
    """
    Function to winsorize a dataframe columns based on the quantile.
    Only upper winsorization is impllemented

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe which columns to winsorize
    columns : Union[str, List[str]]
        Column or a list of columns to winsorize
    quantile : float, optional
        Upper quantile to winsorize the column be, by default 0.99

    Returns
    -------
    pd.DataFrame
        Pandas datarame with the winsorized columns
    """
    input_df[columns] = input_df[columns].clip(upper=input_df[columns].quantile(quantile).tolist(), axis=1)
    return input_df


def get_plotly_shape(x0: float, x1: float, y0: float, y1: float, dash: str = 'solid', opacity: float = 1) -> dict:
    """
    Function to construct a plotly shape objects to be used in the Plotly figure

    Parameters
    ----------
    x0 : float
        Lowest x value
    x1 : float
        Highest x value
    y0 : float
        Lowest y value
    y1 : float
        Highest y value
    dash : str, optional
        Type of the edge line to use, by default 'solid'
    opacity : float, optional
        Coefficient to control the opacity of the figure edges, by default 1

    Returns
    -------
    dict
        Plotly shape dictionary
    """
    return {'type': 'line', 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 
            'line': dict(color='blue', width=1, dash=dash), 'opacity': opacity}


def plot_base_autocorrelation(data: np.ndarray, title: str = None, conf_value: float = 0.05):
    """
    Function to plot value in the ACF, PACF formats.
    Each value correspond to a dot with a vertical line to it. 
    Automatically plots 
    Parameters
    ----------
    data : np.ndarray
        Array with values to plot
    title : str, optional
        Title of the results figure, by default None
    conf_value: float,
        Confidence values (negative and positive) to plot on the figure

    Returns
    -------
        plotly.Figure
    """
    fig = px.scatter(data)
    line_shapes = [
        get_plotly_shape(-1, len(data), conf_value, conf_value, dash='dash', opacity=0.5),
        get_plotly_shape(-1, len(data), -conf_value, -conf_value, dash='dash', opacity=0.5)
    ]
    for ind_d, d in enumerate(data):
        line_shapes.append(
            get_plotly_shape(ind_d, ind_d, min(d, 0), max(d, 0)))
    fig.update_layout(shapes=line_shapes, showlegend=False, xaxis_title='Lags', yaxis_title=None, title = title)
    fig.update_layout(xaxis_range=[-1, len(data)])
    return fig


def plot_acf(input_df: pd.DataFrame, col_name: str, nlags: int = 40, title: str = 'ACF'):
    return plot_base_autocorrelation(acf(input_df[col_name], nlags=nlags), title=title)


def plot_pacf(input_df: pd.DataFrame, col_name: str, nlags: int = 40, title: str = 'PACF'):
    return plot_base_autocorrelation(pacf(input_df[col_name], nlags=nlags), title=title)


class SeasonalityRemover(BasePandasTransformer):
    def __init__(self, target_col: str, date_col: str, season_col: str = 'season', output_col: str = None):
        """
        Class to Remove seasonality pattern using dummy variables
        Firstly - converts date column to the season column (weekday, month, hours, etc...)
        Secondly - encodes season columns as one-hot variable (with no first class just in case)
        Finally - fits linear regression model based on one-hot encoded seasons and calculates deseasoned target values.

        Parameters
        ----------
        target_col : str
            Numeric column which values to adjust
        date_col : str
            Date columns to extract seasons from
        season_col : str, optional
            Column name to store the seasons in, by default 'season'
        output_col : str, optional
            Output column to store deseasoned target values, if None - replaces original target column, by default None
        """
        self.target_col = target_col
        self.date_col = date_col
        self.season_col = season_col
        # If no output colname provided - the target column should be changed inplace
        self.output_col = output_col if output_col else self.target_col
        self.season_model = LinearRegression()
        self.oh_encoder = OneHotEncoder(drop='first')
    
    def extract_season(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y=None):
        df = self.extract_season(x.copy())
        season_matrix = self.oh_encoder.fit_transform(df[[self.season_col]])
        # Linear regression can accept only non-nan values
        mask = x[self.target_col].notna()
        self.season_model.fit(season_matrix[mask, :], x.loc[mask, self.target_col])
        return self
    
    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        x = self.extract_season(x)
        season_matrix = self.oh_encoder.transform(x[[self.season_col]])
        # Linear regression can accept only non-nan values
        mask = x[self.target_col].notna()
        x.loc[mask, self.output_col] = self.season_model.predict(season_matrix[mask, :])
        x[self.output_col] = x[self.target_col] - x[self.output_col]
        return x


class WeekDaySeasonRemover(SeasonalityRemover):
    def extract_season(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df[self.date_col] = pd.to_datetime(input_df[self.date_col])
        input_df.loc[:, self.season_col] = input_df[self.date_col].apply(lambda v: v.weekday)
        input_df.loc[:, self.season_col] = input_df[self.season_col].map({
            0: 'monday',
            1: 'tuesday',
            2: 'wednesday',
            3: 'thursday',
            4: 'friday',
            5: 'saturday',
            6: 'sunday'
        })
        return input_df


class HourSeasonRemover(SeasonalityRemover):
    def extract_season(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df.loc[:, self.season_col] = input_df[self.date_col].apply(lambda v: v.hour)
        return input_df
