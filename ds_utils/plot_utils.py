import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipse_fitted(input_df: pd.DataFrame, one_col_name: str, two_col_name: str, confidence_value: float = 0.95):
    #### TODO - change the plotting engine from matplotlib to Python
    means = input_df[[one_col_name, two_col_name]].mean()
    covariance_matrix = np.cov(input_df[one_col_name], input_df[two_col_name])
    # To make the figure axis-aligned - use eigen values and eigen vectors
    eig_val, eig_vectors = np.linalg.eig(covariance_matrix)
    # The orientation of the ellips could be derived using
    # TODO check why
    angle = np.degrees(np.arctan2(*eig_vectors[:, 0][::-1]))
    confidence_coef = chi2.ppf(confidence_value, 2)
    width = 2 * np.sqrt(confidence_coef * eig_val[0])
    height = 2 * np.sqrt(confidence_coef * eig_val[1])

    # Create a scatter plot of the data
    plt.scatter(input_df[[one_col_name]], input_df[[two_col_name]], color='blue', alpha=0.5)
    ellipse = Ellipse((means[0], means[1]), width, height, angle, alpha=0.5, color='red')

    # Add the ellipse to the plot
    ax = plt.gca()
    ax.add_patch(ellipse)
    # Set the aspect ratio to 'equal' for a circular ellipse
    ax.set_aspect('equal')
    # Set the plot limits
    plt.xlim(-5, 10)
    plt.ylim(-5, 10)
    # Display the plot
    plt.show()
