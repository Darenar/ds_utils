from typing import Tuple, Union, List, Callable

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class EllipseShape:
    x_center: float
    y_center: float
    width: float
    height: float
    angle: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.x_center, self.y_center
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """
        Checks if the ellipse contains the provided point
        Parameters
        ----------
        point : tuple of float
            Point coordinates

        Returns
        -------
        Boolean
            True if the point is inside the ellipse
        """
        # Convert the angle to radians
        angle_rad = np.deg2rad(self.angle)

        # Calculate the rotated coordinates
        x_rot = (point[0] - self.x_center) * np.cos(angle_rad) + (point[1] - self.y_center) * np.sin(angle_rad)
        y_rot = -(point[0] - self.x_center) * np.sin(angle_rad) + (point[1] - self.y_center) * np.cos(angle_rad)

        # Calculate the values to check against the ellipse equation
        x_val = (x_rot / (self.width / 2)) ** 2
        y_val = (y_rot / (self.height / 2)) ** 2

        if x_val + y_val <= 1:
            return True
        return False

    @classmethod
    def fit(cls, input_df: pd.DataFrame, one_col_name: str, two_col_name: str, proportion: float = 0.95) -> 'EllipseShape':
        """
        Function to fit an ellipse that would separate given proportion of data samples from outliers.
        Visually - on the graph with sample points fits the ellipse that will include at least *proportion* of elements

        Parameters
        ----------
        input_df : pd.DataFrame
            Input dataframe
        one_col_name : str
            First column name
        two_col_name : str
            Second column name
        proportion : float, optional
            Percentage of samples to be inside the ellipse, by default 0.95

        Returns
        -------
        EllipseShape
            Fitted ellipse shape
        """
        means = input_df[[one_col_name, two_col_name]].mean()
        covariance_matrix = np.cov(input_df[one_col_name], input_df[two_col_name])
        # To make the figure axis-aligned - use eigen values and eigen vectors
        eig_val, eig_vectors = np.linalg.eig(covariance_matrix)
        # The orientation of the ellips could be derived using
        angle = np.degrees(np.arctan2(*eig_vectors[:, 0][::-1]))
        confidence_coef = stats.chi2.ppf(proportion, 2)
        return cls(
            x_center=means[0],
            y_center=means[1],
            width=2 * np.sqrt(confidence_coef * eig_val[0]),
            height=2 * np.sqrt(confidence_coef * eig_val[1]),
            angle=angle)
    
    def get_boundary_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates for columns that lie on the ellipse boundaries
        Returns
        -------
        Tuple of arrays
            X and Y coordinates
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        angle_rad = np.deg2rad(self.angle)
        x = self.x_center + (self.width / 2) * np.cos(theta) * np.cos(angle_rad) - (self.height / 2) * np.sin(theta) * np.sin(angle_rad)
        y = self.y_center + (self.width / 2) * np.cos(theta) * np.sin(angle_rad) + (self.height / 2) * np.sin(theta) * np.cos(angle_rad)
        return x, y


def winsorize(input_df: pd.DataFrame, columns: Union[str, List[str]], upper_quantile: float = 0.99, lower_quantile: float = 0.01) -> pd.DataFrame:
    """
    Function to winsorize a dataframe columns based on the quantile.
    Only upper winsorization is impllemented

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe which columns to winsorize
    columns : Union[str, List[str]]
        Column or a list of columns to winsorize
    upper_quantile : float, optional
        Upper quantile to winsorize the column be, by default 0.99
    lower_quantile : float, optional
        Lower quantile to winsorize the column be, by default 0.01

    Returns
    -------
    pd.DataFrame
        Pandas datarame with the winsorized columns
    """
    input_df[columns] = input_df[columns].clip(upper=input_df[columns].quantile(upper_quantile).tolist(), axis=1)
    input_df[columns] = input_df[columns].clip(lower=input_df[columns].quantile(lower_quantile).tolist(), axis=1)
    return input_df


def append_empty_dates(input_df: pd.DataFrame, date_col: str, fill_cols: Union[str, List[str]] = None, 
                       min_date: str = None, max_date: str = None, id_col: str = None) -> pd.DataFrame:
    """
    Append missing dates observations to the data frame.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe with dates
    date_col : str
        Column with dates
    fill_cols : Union[str, List[str]], optional
        List of columns to fill the values for, by default None
    min_date: str
        Minimum date to fill from, by default will use the lowest in the dataframe
    max_date: str
        Maximum date to fill from, by default will use the highest in the dataframe
    id_col: str
        ID column to group by. 

    Returns
    -------
    pd.DataFrame
        DataFrame with missing dates appended.
    """
    if id_col:
        # If no fill cols provided, make sure ID col is added there.
        if not fill_cols:
            fill_cols = [id_col]
        elif id_col not in fill_cols:
            fill_cols.append(id_col)
        # Apply the same function to each of the group while setting ID col to None.
        return input_df.groupby(id_col).apply(
            lambda sub_df: append_empty_dates(
                sub_df, 
                date_col=date_col,
                fill_cols=fill_cols,
                min_date=min_date,
                max_date=max_date, 
                id_col=None)).reset_index(drop=True)
    # Covert column to date format
    input_df[date_col] = pd.to_datetime(input_df[date_col]).dt.date
    # If min or max date not provided - use the ones from the dataframe itself.
    min_date = min_date if min_date else input_df[date_col].min()
    max_date = max_date if max_date else input_df[date_col].max()
    date_range = pd.date_range(start=min_date, end=max_date)
    # Reindex by new date range
    input_df =  input_df.set_index(date_col).reindex(date_range).reset_index().rename({'index': date_col}, axis=1)
    if fill_cols:
        # Fill nans for the specified columns only
        input_df.loc[:, fill_cols] = input_df[fill_cols].ffill().bfill()
    return input_df


def rolling_apply(input_df: pd.DataFrame, func: Callable, date_col: str, window: int, feature_cols: List[str], 
                  prefix: str, exclude_current: bool=False, id_col: str = None) -> pd.DataFrame:
    """
    Function to apply func in a rolling fashion for the specified columns
    Parameters
    ----------
    input_df : pd.DataFrame
        Input data frame with features
    func : callable
        function to aply in a rolling window fashion
    date_col : str
        Date column
    window : int
        Window size to use in calculations
    feature_cols : List[str]
        Features to apply rolling func for
    exclude_current : bool, optional
        Boolean of whether the current state should be excluded from the calculations, by default False
    id_col : str, optional
        If provided, the rolling func values will be caclulated on each group separately, by default None
    prefix: str
        Prefix to append to feature rolling func values columns

    Returns
    -------
    pd.DataFrame
        Output dataframe with calculated rolling func values per feature.
    """
    if id_col:
        roll_df = input_df.groupby(id_col).apply(
            lambda sub_df: rolling_apply(
                input_df=sub_df, 
                func=func,
                date_col=date_col,
                window=window,
                feature_cols=feature_cols,
                exclude_current=exclude_current,
                id_col=None,
                prefix=prefix
            ))
        return input_df.join(roll_df)
    # Sort by dates
    input_df.sort_values(date_col, inplace=True)
    # Apply rolling func for the specified columns
    return input_df.rolling(
        window=window, 
        min_periods=1,
        closed='right' if not exclude_current else 'left')[feature_cols].apply(func).add_prefix(prefix).copy()
