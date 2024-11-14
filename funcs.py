# Auxiliary functions

import pandas as pd
import numpy as np
from typing import List
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import chi2, chi2_contingency
from typing import Union

# Wranggling data #######################################################################################################

def load_train_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a text file into a Pandas DataFrame, where each record
    is represented by a row and specific columns are defined.

    The function expects a text file with data entries separated by records
    labeled as "Record N" (e.g., "Record 1", "Record 2"). Each record contains 
    multiple lines of data in a specific order.

    Parameters:
    ----------
    filepath : str
        The path to the text file containing the data.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns: 'id', 'durationOfStay', 'gender', 'age',
        'kids', 'destinationCode', 'acomType'.
    """

    columns: List[str] = ["id", "durationOfStay", "gender", "age", "kids", "destinationCode", "acomType"]
    data: List[List[str]] = []
    
    with open(filepath, 'r') as file:
        record = []
        for line in file:
            line = line.strip()
            if "Record" in line:
                if record:
                    data.append(record)
                record = [] 
            elif line:
                entry = line.replace("[1]", "").strip()
                entry = pd.NA if entry == "<NA>" else entry
                record.append(entry)
        if record:
            data.append(record)

    return pd.DataFrame(data, columns=columns)


def random_chi_squared_impute(series: pd.Series) -> pd.Series:
    """
    Impute missing values in a series using random values drawn from a chi-squared distribution.

    Parameters:
    ----------
    series : pd.Series
        The Pandas Series with missing values to be imputed. The distribution of existing values 
        should resemble a chi-squared distribution for this method to be appropriate.

    Returns:
    -------
    pd.Series
        The series with missing values imputed based on a fitted chi-squared distribution.
    """
    df_fit = chi2.fit(series.dropna())[0]

    n_missing = series.isna().sum()

    random_values = np.random.chisquare(df_fit, n_missing)

    series.loc[series.isna()] = random_values
    
    return series

## Analysis ############################################################################################################

def cramers_v(x: Union[pd.Series, list], y: Union[pd.Series, list]) -> float:
    """
    Calculate Cramér's V statistic for categorical-categorical association.
    
    Cramér's V is used to measure the association between two categorical variables,
    with values ranging from 0 (no association) to 1 (perfect association).

    Parameters:
    - x (Union[pd.Series, list]): The first categorical variable.
    - y (Union[pd.Series, list]): The second categorical variable.

    Returns:
    - float: Cramér's V statistic, representing the association between `x` and `y`.
    """
    contingency_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def create_cramers_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Cramér's V correlation matrix for a DataFrame with categorical columns.

    Parameters:
    - df (pd.DataFrame): A DataFrame where each column represents a categorical variable.

    Returns:
    - pd.DataFrame: A DataFrame representing the Cramér's V correlation matrix.
    """
    columns = df.columns
    n = len(columns)
    correlation_matrix = pd.DataFrame(np.zeros((n, n)), columns=columns, index=columns)
    for col1 in columns:
        for col2 in columns:
            correlation_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return correlation_matrix

## Plotting #############################################################################################################

def interactive_barplot(df: pd.DataFrame, column: str):
    """
    Bar plot with plotly express

    Parameters:
    ----------
    df: pd.Dataframe
        Data
    
    column: str 
        Desired column name
    """

    if df[column].dtype == 'object':
        fig = px.histogram(df.fillna({column: "Unknown"}), x=column, color=column, barmode='relative')
    
    else:
        fig = ff.create_distplot([df[column].dropna().tolist()], [column], show_hist=True, show_rug=False)
        
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False
    )

    fig.show()


def interactive_grouped_bar_plot(df: pd.DataFrame, column: str, group: str):
    """
        Function to generate interactive grouped barplot


    Args:
        df (pd.DataFrame): data
        column (str): column name
        group (str): grouping column name
    """
    if df[column].dtype == 'object':
        fig = px.histogram(df.fillna({column: "Unknown"}), x=column, color=group, barmode='stack')
    
    else:
        grouped_data = [df[df[group] == label][column].dropna().tolist() for label in df[group].unique()]
        group_labels = df[group].unique().astype(str)
        fig = ff.create_distplot(grouped_data, group_labels, show_hist=True, show_rug=False)

        
    fig.update_layout(
    title=f"Distribution of {column} vs {group}",
    xaxis_title=column,
    yaxis_title="",
    showlegend=True
    )

    fig.show()


def interactive_stacked_plot(df: pd.DataFrame, column: str, group: str) -> None:
    """
    Generates an interactive stacked bar plot showing the relative (percentage)
    and absolute (count) frequency distribution of a specified categorical column,
    grouped by another categorical column. Each stack sums to 100% within each 
    primary category on the x-axis.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be plotted.
    - column (str): The main categorical column to display along the x-axis.
    - group (str): The categorical column used to color-code the stacks in the plot.

    Returns:
    - None: Displays the plot directly.

    Example:
    interactive_stacked_plot(df, column='destinationCode', group='acomType')
    """
    percentage_data = (
        df
        .fillna({group: "Unknown"})
        .groupby([column, group])
        .size()
        .reset_index(name='count')
    )
    
    percentage_data['percent'] = (
        percentage_data.groupby(column)['count']
        .transform(lambda x: 100 * x / x.sum())
    )

    fig = px.bar(
        percentage_data, 
        x=column, 
        y='percent', 
        color=group, 
        barmode='stack', 
        text='count' 
    )

    fig.update_traces(texttemplate='%{text} (%{y:.2f}%)') 

    fig.update_layout(
        title=f"Frecuencias - {column} por {group}",
        xaxis_title=column,
        yaxis_title="Frequency",
        legend_title=group
    )
    
    fig.show()
