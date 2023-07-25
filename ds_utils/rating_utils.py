from typing import List, Union

import pandas as pd
import numpy as np

from .base import BasePandasTransformer


EPSILON = 1e-10
LABELS = ['D', 'C', 'B', 'A']
SIGNED_LABELS = ['D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']


def binarize_series_by_labels(input_series: pd.Series, ordered_labels: List[Union[str, int, float]], 
                              min_val: float = 0., max_val: float = 100.) -> pd.Series:
    """
    Function to equally-binarize values by the provided 

    Parameters
    ----------
    input_series : pd.Series
        Pandas series with scores to binarize
    ordered_labels : list
        List of labels in ascending order to by assigned to different bins
    min_val : float, optional
        Min value used to derive a bin size, by default 0.
    max_val : float, optional
        Max value used to derive a bin size, by default 100.

    Returns
    -------
    pd.Series
        Pandas series with binarized values
    """
    n_bins = len(ordered_labels)
    bin_thresholds = np.arange(min_val-EPSILON, max_val+EPSILON, (max_val-min_val) / n_bins)
    return pd.cut(
        input_series, 
        bins=bin_thresholds,
        labels=ordered_labels
    )


class RatingSteadyStateHandler(BasePandasTransformer):
    """
    Function to calculate steady state of the rating system. 

    Parameters
    ----------
    id_col : str
        ID column to represent different entities in the set
    time_col : str
        Time column to be used to sort the entries per ID
    feature_col : str
        Particular feature (rating) to calculate the steady state for.
    """
    PREV_PREFIX: str = 'Prev'
    def __init__(self, id_col: str, time_col: str, feature_col: str):
        self.id_col = id_col
        self.time_col = time_col
        self.feature_col = feature_col
        self.transition_counts_matrix = None
        self.transition_prob_matrix = None
        self.init_state = None
        self.steady_state = None
    
    def fit(self, x: pd.DataFrame, y=None):
        self.fit_transition_prob_matrix(x)
        self.fit_steady_state()
        return self


    def fit_transition_prob_matrix(self, x: pd.DataFrame):
        # Sort data frame by ID and Time
        sorted_x = x.sort_values([self.id_col, self.time_col]).copy()
        sorted_x = sorted_x[[self.id_col, self.time_col, self.feature_col]]
        # Add a new column that has previous rating per time-period per id
        sorted_x.loc[:, f"{self.PREV_PREFIX}{self.feature_col}"] = sorted_x.groupby(self.id_col)[self.feature_col].shift(1)
        # Calculate number of transitions from state to state and unstack to have a count-transition df
        self.transition_counts_matrix = sorted_x.groupby([
            f"{self.PREV_PREFIX}{self.feature_col}", 
            self.feature_col])[self.time_col].count().unstack()
        # Store initial state of the Markov Chain
        self.init_state = (self.transition_counts_matrix.sum(axis=1) / self.transition_counts_matrix.sum().sum()).values
        
        # Convert counts to probabilities. If all elements are 0 - then assign them a equally weighted probability.
        self.transition_prob_matrix = self.transition_counts_matrix.apply(lambda row: 
            row / row.sum() if row.sum() != 0 
            else 1/len(row), axis=1)
    
    def fit_steady_state(self):
        # Calculate steady state using the eigenvalue-eigenvector approach
        eigen_vals, eigen_vectors = np.linalg.eig(self.transition_prob_matrix.T)
        # Create a mask that specifies the eigen value with the magnitude closest to 1
        value_mask = np.isclose(eigen_vals, 1)
        if not sum(value_mask):
            raise ValueError("There is no Eigen Value with magnitude close to 1")
        
        # Select Eigen vector that is associated with the highest eigen value
        selected_eigen_vector = eigen_vectors[:, value_mask].flatten()
        # Normalize vector to represent marginal probabilities + make sure to take real  part, as there could be complex numbers
        self.steady_state = (selected_eigen_vector / selected_eigen_vector.sum()).real
    
    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.steady_state

    def get_time_states(self, n_steps: int) -> pd.DataFrame:
        """
        Provides a data frame with the states of the Markov Chain over n_steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to calculate the states for

        Returns
        -------
        pd.DataFrame
            DataFrame with the states per each period
        """
        if self.steady_state is None:
            raise ValueError(f"No measure has been fit. Run fit() function first.")
        
        time_states = [self.init_state]
        for _ in range(n_steps):
            prev_state = time_states[-1]
            time_states.append(
                np.dot(self.transition_prob_matrix.T, prev_state)
            )
        return pd.DataFrame(time_states, columns=self.transition_prob_matrix.columns)
