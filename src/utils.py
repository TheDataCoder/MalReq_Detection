import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typing

from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, normalize
from ua_parser import user_agent_parser


def parse_ua(ua):
    p = user_agent_parser.Parse(ua)
    return [p.get("os").get("family"), p.get("user_agent").get("family")]


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=4, weights="distance")
    imputer_fit = imputer.fit(df.values)
    joblib.dump(imputer_fit, "knn_imputer.pt")
    df = pd.DataFrame(imputer.fit_transform(df.values), columns=df.columns)
    return df


def encode_categorical_variables(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
    return df


def scale_values_in_df(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(normalize(df), columns=df.columns)
    return df


def corr_matrix(df: pd.DataFrame, func: typing.Any) -> pd.DataFrame:
    """
    Builds the correlation matrix
        :param df: pandas dataframe
        :param func: type of correlation function to use
    :return: pandas dataframe with correlation values
    """
    columns = df.columns
    K = df.shape[1]
    if K <= 1:
        return pd.DataFrame()
    else:
        corr_array = np.eye(K)

        for i in range(K):
            for j in range(K):
                if i <= j:
                    continue
                c = func(df[columns[i]], df[columns[j]])
                corr_array[i, j] = c
                corr_array[j, i] = c
        return pd.DataFrame(data=corr_array, columns=columns, index=columns)


def calculate_correlations(df: pd.DataFrame, num_for_corr: list, cat_for_corr: list, kind: str) -> pd.DataFrame:
    """
    Calculates different kids of correlation values
        :param df: pandas dataframe
        :param num_for_corr: numeric features for correlation
        :param cat_for_corr: categorical features for correlation
        :param kind: one of: 'pearson', 'spearman', 'kendall', cramer
    :return: correlation matrix
    """
    if kind == "pearson":
        return df[num_for_corr].corr("pearson")
    elif kind == "spearman":
        return df[num_for_corr].corr("spearman")
    elif kind == "kendall":
        return df[num_for_corr].corr("kendall")
    elif kind == "cramer":
        return corr_matrix(df[cat_for_corr], cramer)


def cramer(x: pd.Series, y: pd.Series) -> float:
    """
    Calculates Cramér's V values
        :param x: pd.series: feature one
        :param y: pd.series: feature two
    :returns: numpy array
    """
    arr = pd.crosstab(x, y).values
    chi2_stat = chi2_contingency(arr, correction=False)
    phi2 = chi2_stat[0] / arr.sum()
    n_rows, n_cols = arr.shape
    if min(n_cols - 1, n_rows - 1) == 0:
        value = np.nan
    else:
        value = np.sqrt(phi2 / min(n_cols - 1, n_rows - 1))

    return value


def plot_correlation_figure(df: pd.DataFrame, num_features: list, cat_features: list, kind: list) -> None:
    """
    Plot correlation matrix's of different types
        :param df - pandas dataframe
        :param num_features - integer, number of features
        :param cat_features - list of categorical features
        :param kind: "pearson", "spearman", "kendall", "cramer"
    :returns: Plotly plot wtih correlation matrix
    """
    df = df.copy()

    for kind in kind:
        correlation = calculate_correlations(df, num_features, cat_features, kind=kind)
        columns = correlation.columns
        subplot_titles = ["Correlation - " + kind]
        fig = make_subplots(rows=1, cols=1, subplot_titles=subplot_titles, shared_yaxes=True)

        if len(columns) < 15:
            text = np.round(correlation, 2).astype(str)
            texttemplate = "%{text}"
        else:
            text = None
            texttemplate = None

        trace = go.Heatmap(
            z=correlation, x=columns, y=columns, text=text, texttemplate=texttemplate, coloraxis="coloraxis"
        )
        fig.append_trace(trace, 1, 1)
        fig.update_layout(coloraxis={"colorscale": "RdBu"}, width=800, height=800)
        fig.show()
