import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import acf, pacf
import plotly.graph_objects as go
import plotly.express as px

from ..preprocessing_utils import EllipseShape


def get_plotly_shape(x0: float, x1: float, y0: float, y1: float, dash: str = 'solid', opacity: float = 1, color: str='blue') -> dict:
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
    color: str, optiona;
        Color of the resulted shape, by default blue

    Returns
    -------
    dict
        Plotly shape dictionary
    """
    return {'type': 'line', 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 
            'line': dict(color=color, width=1, dash=dash), 'opacity': opacity}


def plot_scatter_with_ellipse(input_df: pd.DataFrame, one_col_name: str, two_col_name: str, 
                              proportion: float = 0.9, plot_ols: bool = True, suffix: str = '') -> go.Figure:
    """
    Function to create a scatter plot with an ellipse fitted to separate the main proportion of data from outliers.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe
    one_col_name : str
        Feature one
    two_col_name : str
        Feature two
    proportion : float, optional
        Proportion of main data to separate from outliers using an ellipse, by default 0.9
    plot_ols : bool, optional
        Whether to plot fitted OLS model or not, by default True
    suffix : str, optional
        Suffix to add to the titles of the resulted graph, by default ''

    Returns
    -------
    go.Figure
        Plotly scatter plot with a fitted ellipse
    """
    # Estimate an ellipse
    title=f'Ellipse for {proportion} proportion {suffix}'
    ellipse = EllipseShape.fit(input_df, one_col_name=one_col_name, two_col_name=two_col_name, proportion=proportion)
    boundary_x_array, boundary_y_array = ellipse.get_boundary_points()
    
    # Prepare figure layout
    max_value = max(input_df[one_col_name].max(), input_df[two_col_name].max())
    min_value = max(input_df[one_col_name].min(), input_df[two_col_name].min())
    layout = go.Layout(
        title=title,
        xaxis=dict(title=one_col_name, 
            range=[min_value, max_value * 1.05]),
        yaxis=dict(title=two_col_name, range=[min_value, max_value * 1.05]),
        showlegend=False
    )

    # Get scatter plot with a fitted ellipse on it
    fig = go.Figure(
        data=[
            # get ellipse object
            go.Scatter(
                x=boundary_x_array,
                y=boundary_y_array,
                mode='lines',
                line=dict(color='blue', width=1),
                name='Ellipse',
                showlegend = False
            ),
            # plot all samples
            go.Scatter(x=input_df[one_col_name].tolist(), 
                       y=input_df[two_col_name].tolist(), 
                       mode='markers', 
                       # Make all inner points as blue markers and all outliers as red markers
                       marker= {
                           'color': ['blue' if ellipse.contains(v) else 'red' for v in input_df[[one_col_name, two_col_name]].values],
                           },
                       showlegend = False)
            ], layout=layout)
    
    if plot_ols:
        # Select only those samples that do fall into the ellipse
        inner_points = [v for v in input_df[[one_col_name, two_col_name]].values if ellipse.contains(v)]
        
        # Calculate OLS estimates
        ols_slope, ols_intercept, _, ols_p_value, _ = stats.linregress(*zip(*np.vstack(inner_points)))
        
        # Add a line to the plot
        fig.add_shape(**get_plotly_shape(
            min_value, max_value, 
            ols_intercept + (min_value*ols_slope), ols_intercept + (max_value*ols_slope), 
            color='blue', dash='dash'))
        fig.update_layout(title=f'Ellipse for {proportion} with OLS slope {round(ols_slope, 2)} (p-value {round(ols_p_value,2)}) {suffix}')
    return fig


def plot_spread_graph_with_shaded_areas(input_df: pd.DataFrame, column_one: str, column_two: str, title: str = '') -> go.Figure:
    """
    Function to plot a spread graph between two features, while also shading the area between them:
    1. When feature 1 is greater than feature 2 - shaded area is green
    2. When feature 1 is lower than feature 2 - shaded area is reds
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Input data frame with the features
    column_one : str
        Feature one
    column_two : str
        Feature two
    title : str, optional
        title of the resulted graph, by default ''

    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Get a scatter plot for feature one
    trace_one = go.Scatter(
        x=input_df.index,
        y=input_df[column_one],
        name=column_one,
        mode='lines',
        line=dict(color='rgb(102,166,30)')
    )
    # Get a scatter plot for feature two
    trace_two = go.Scatter(
        x=input_df.index,
        y=input_df[column_two],
        name=column_two,
        mode='lines',
        line=dict(color='rgb(204,80,62)')
    )
    # Get a scatter for the green shaded area
    trace_fill_one = go.Scatter(
        x=input_df.index,
        y=input_df[[column_one, column_two]].max(axis=1),
        fill='tonext',
        mode='lines',
        fillcolor='rgba(10,255,10,0.3)',
        line=dict(width=0),
        showlegend=False,
    )
    # Get a scatter for the red shaded area
    trace_fill_two = go.Scatter(
        x=input_df.index,
        y=input_df[[column_one, column_two]].min(axis=1),
        fill='tonextx',
        mode='lines',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(width=0),
        showlegend=False,
    )
    # Update layout  
    layout = go.Layout(
        title=title
    )
    fig = go.Figure(data=[
        trace_one, 
        trace_two, 
        trace_fill_one,
        trace_fill_two
        ], layout=layout
    )
    return fig
    

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

