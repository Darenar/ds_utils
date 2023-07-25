import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from .base import BasePandasTransformer


def get_corr_by_rolling_window(input_df: pd.DataFrame, column_one: str, column_two: str, window_size: int, method: str = 'spearman'):
    """
    Function to calculate rolling correlaton between two columns. 
    Could not be done through pandas rolling functionality, as there is a bug with non-pearson method calcualtion. Thus, do it manually.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe with time-series
    column_one : str
        Feature one
    column_two : str
        Feature two
    window_size : int
        Window size to roll over
    method : str, optional
        Type of correlation to use according to pandas corr, by default 'spearman'

    Returns
    -------
    pd.DataFrame
        Rolling correlation results as a pandas dataframe
    """
    result_list = list()
    for i in range(input_df.shape[0]):
        # Mark as none the periods with low number of observations
        if i < window_size-1:
            result_list.append(None)
            continue
        start_index = i - window_size + 1
        result_list.append(input_df.iloc[start_index:i][column_one].corr(
            input_df.iloc[start_index:i][column_two], method=method
        ))
    return pd.DataFrame(result_list, index=input_df.index, columns=[method])


def get_corr_by_shift(input_df: pd.DataFrame, column_one: str, column_two: str, max_abs_shift: int = 30, method: str = 'spearman') -> pd.DataFrame:
    """
    Function to calculate correlation values for different shift values. Similar to MatLab's XCORR functonality.
    Negative shift means that the current ColumnOne affects future ColumnTwo.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe with time-series
    column_one : str
        Feature one
    column_two : str
        Feature two
    max_abs_shift : int, optional
        Maximum absolute shift to be calculated from both sides, by default 30
    method : str, optional
        Type of correlation to use according to pandas corr, by default 'spearman'

    Returns
    -------
    pd.DataFrame
        Correlation results for different shift values
    """
    result_list = list()
    for s in range(-max_abs_shift, max_abs_shift, 1):
        shift_df = input_df[[column_one, column_two]].copy()
        shift_df.loc[:, column_two] = shift_df[column_two].shift(s)
        corr_value = shift_df[column_one].corr(shift_df[column_two], method=method)
        result_list.append((s, corr_value))
    return pd.DataFrame(result_list, columns=['shift', f'{method}_corr'])


def rolling_optimal_shift(input_df: pd.DataFrame, column_one: str, column_two: str, window_size: int, method: str = 'spearman', max_abs_shift: int = 30):
    """
    Calculate optimal shift that maximizes the correlation measure in a rollng window fashion.
    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe with time-series
    column_one : str
        Feature one
    column_two : str
        Feature two
    window_size : int
        Window size to roll over
    method : str, optional
        Type of correlation to use according to pandas corr, by default 'spearman'
    max_abs_shift : int, optional
        Maximum absolute shift to be calculated from both sides, by default 30

    Returns
    -------
    pd.DataFrame
        Rolling correlation results with an optimal shift
    """
    result_list = list()
    for i in range(input_df.shape[0]):
        if i < window_size-1:
            result_list.append(None)
            continue
        start_index = i - window_size + 1
        shift_df = get_corr_by_shift(input_df=input_df.iloc[start_index:i], column_one=column_one, 
                                     column_two=column_two, max_abs_shift=max_abs_shift, 
                                     method=method)
        result_list.append(
            shift_df.sort_values(f'{method}_corr', ascending=False)['shift'].iloc[0]
        )
    return pd.DataFrame(result_list, index=input_df.index, columns=[method])


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
    """
    Class to remove WeekDay seasonality from the feature
    """
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
    """
    Class to remove hour seasonality from the feature
    """
    def extract_season(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df.loc[:, self.season_col] = input_df[self.date_col].apply(lambda v: v.hour)
        return input_df
