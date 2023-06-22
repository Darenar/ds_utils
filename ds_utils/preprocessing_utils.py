from typing import Tuple, Union, List

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