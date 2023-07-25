import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


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
    return norm.ppf( (1-critical_value) / 2) * sigma * np.sqrt(np.arange(1, n_timesteps + 1))


def plot_cumsum(input_series: pd.Series):
    processed_series = (input_series - input_series.mean()) / (input_series.std())
    conf_level_series = get_cumulitative_confidence_level(processed_series.std(), processed_series.shape[0])
    
    processed_series = processed_series.to_frame(input_series.name)
    processed_series.loc[:, 'upper_bound'] = conf_level_series
    processed_series.loc[:, 'lower_bound'] = - conf_level_series

    trace_main = go.Scatter(x = processed_series.index, y = processed_series[input_series.name].cumsum(), name=input_series.name)
    trace_upper = go.Scatter(
        x = processed_series.index, 
        y = processed_series['upper_bound'], 
        showlegend=False,
        line=dict(color='red', width=1)
    )
    trace_lower = go.Scatter(
        x = processed_series.index, 
        y = processed_series['lower_bound'], 
        showlegend=False,
        line=dict(color='red', width=1)
    )
    layout = go.Layout(
        title=f'{input_series.name} CumSum plot with confidence levels',
        xaxis=dict(title=input_series.index.name),
        yaxis=dict(title='Normalized CumSum'))
    
    fig = go.Figure(data=[
        trace_main, trace_upper, trace_lower
        ], layout=layout
    )
    return fig


def plot_cumsum_monte_carlo(n_timesteps: int, n_iter: int, mean: float = 0., sigma: float = 1.):
    conf_level_series = get_cumulitative_confidence_level(sigma, n_timesteps)
    
    processed_series = pd.DataFrame({'random_series': np.random.normal(mean, sigma, size=n_timesteps)})
    processed_series.loc[:, 'upper_bound'] = conf_level_series
    processed_series.loc[:, 'lower_bound'] = - conf_level_series

    list_of_trial_traces = list()
    for i in range(n_iter):
        list_of_trial_traces.append(
            go.Scatter(
                x = processed_series.index, 
                y = np.random.normal(mean, sigma, size=n_timesteps).cumsum(), 
                showlegend=False)
        )
    
    trace_upper = go.Scatter(
        x = processed_series.index, 
        y = processed_series['upper_bound'], 
        showlegend=False, 
        line=dict(color='red', width=3)
    )
    trace_lower = go.Scatter(
        x = processed_series.index, 
        y = processed_series['lower_bound'], 
        showlegend=False,
        line=dict(color='red', width=3)
    )
    layout = go.Layout(
        title=f'Monte-Carlo CumSum plot with confidence levels',
        yaxis=dict(title='Normalized CumSum'))
    
    fig = go.Figure(data=list_of_trial_traces + [trace_upper, trace_lower], layout=layout
    )
    return fig