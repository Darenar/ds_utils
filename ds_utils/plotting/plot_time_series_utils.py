import numpy as np
from scipy.stats import norm
import cufflinks as cf

cf.go_offline()


def get_cumulitative_confidence_level(sigma: float, n_timesteps: int, critical_value: float = 0.95) -> np.ndarray:
    """
    Get confidence level for a cumulative sum plot. 
    Parameters
    ----------
    sigma : float
        Standard Deviation of the time-series before the CumSum
    n_timesteps : int
        Number of timestamp to generate confidence levels for
    critical_value : float, optional
        Critical value from normal distribution, by default 0.95

    Returns
    -------
    Array 
        Confidence level array
    """
    return norm.ppf((1-critical_value) / 2) * sigma * np.sqrt(np.arange(1, n_timesteps + 1))


def get_permutation_cumsum_plot(input_df, col_name: str, n_perm: int = 100, main_col_line_width: int = 4):
    """
    Plots permutation cumsum plot for a provided feature. In essence:
    1. Standardizes the series
    2. Generate n_perm random permutation
    3. Calculates cumulative sum for the original series and for all the permutations
    4. Plots all together

    Parameters
    ----------
    input_df : pd.DataFrame
        Input data frame
    col_name : str
        Feature name to build a permutation plot for
    n_perm : str, optional
        Number of permutations to generate and plot, by default 100
    main_col_line_width : int, optional
        Width of the main line to plot, by default 4

    Returns
    -------
    Plotly figure   
    """
    # Standardize the dataframe
    norm_df = (input_df[col_name] - input_df[col_name].mean()) / input_df[col_name].std()
    norm_df = norm_df.to_frame()
    
    # Create n_perm random permutations
    for i in range(n_perm):
        norm_df.loc[:, f"permutation_{i}"] = norm_df[col_name].sample(norm_df.shape[0]).values.flatten()
    norm_df = norm_df.cumsum()
    fig = norm_df.iplot(showlegend=False, title="Monte-Carlo CumSum plot", yTitle='Normalized CumSum', asFigure=True)

    for trace in fig.data:
        if trace.name == col_name:
            trace.line.width = main_col_line_width
    return fig


def get_cumsum_plot(input_df, col_name: str, divide_by_t: bool=True):
    """
    Plot a cumulative sum plot with the cumulative confidence levels

    Parameters
    ----------
    input_df : pd.DataFrame
        Input data frame
    col_name : str
        Feature name to build a permutation plot for
    divide_by_t : bool, optional
        Whether to divide cumulitative sum by the SQRT of time-periods, by default True

    Returns
    -------
    Plotly figure
    """
    # Standardize the dataframe
    norm_df = (input_df[col_name] - input_df[col_name].mean()) / input_df[col_name].std()
    norm_df = norm_df.to_frame()

    # Get 95% confidence band by 1.96 * STD (which is = 1 because of standardization) * SQRT(Time)
    confidence_band = get_cumulitative_confidence_level(norm_df[col_name].std(), norm_df.shape[0])
    norm_df = norm_df.cumsum()
    if divide_by_t:
        norm_df.loc[:, col_name] = norm_df[col_name] / np.sqrt(np.arange(1, norm_df.shape[0]+1))
    norm_df.loc[:, 'upper'] = confidence_band
    norm_df.loc[:, 'lower'] = - confidence_band
    return norm_df.iplot(
        title=f'{col_name} CumSum plot with confidence levels',
        xTitle=col_name, yTitle='Normalized CumSum', asFigure=True)
