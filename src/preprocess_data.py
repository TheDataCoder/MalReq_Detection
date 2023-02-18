import joblib
import pandas as pd

from dataclasses import dataclass
from functools import reduce
from sklearn.preprocessing import StandardScaler
from utils import parse_ua


@dataclass
class DataProcessor:
    """
    The Data class to preprocess your data into prediction-ready format.
    """

    @staticmethod
    def __get_browsers_onehot(df: pd.DataFrame) -> pd.DataFrame:
        browser_list = ["bot", "curl", "phantom_js", "openvas", "python", "java", "riddler", "apache", "scrapy" "other"]
        for browser in browser_list:
            df[f"BROWSER_{str.upper(browser)}"] = df["BROWSER"].map(lambda x: 1 if browser in str.lower(x) else 0)
        df["BROWSER_NON_SPEC"] = df["BROWSER"].map(lambda x: 1 if x in browser_list else 0)
        return df

    @staticmethod
    def __get_matched_variable_onehot(df: pd.DataFrame) -> pd.DataFrame:
        variables = ["document.cookie"]
        df["MATCHED_VARIABLE_VALUE"] = df["MATCHED_VARIABLE_VALUE"].astype(str)
        for variable in variables:
            df[f"MATCHED_VARIABLE_{str.upper(variable)}"] = df["MATCHED_VARIABLE_VALUE"].map(
                lambda x: 100 if variable in str.lower(x) else 0
            )
        df["MATCHED_VARIABLE_PWD"] = df["MATCHED_VARIABLE_VALUE"].map(
            lambda x: 30 if "passwd" in str.lower(x) or "pwd" in str.lower(x) else 0
        )
        df["MATCHED_VARIABLE_REDIRECT"] = df["MATCHED_VARIABLE_VALUE"].map(
            lambda x: 20 if "redirect" in str.lower(x) else 0
        )
        return df

    def process(self, df):
        """
        Dataframe feature processor
            :param df: pandas DataFrame to process
            :return: processed pandas DataFrame
        """

        df = df[~df["CLIENT_IP"].isna()]
        df = df[df["CLIENT_IP"].str.match(r"^\d+\.\d+\.\d+\.\d+\.*")]
        df["CLIENT_USERAGENT"] = df["CLIENT_USERAGENT"].fillna("NaN")
        df["MATCHED_VARIABLE_SRC"] = df["MATCHED_VARIABLE_SRC"].astype(str)
        df["MATCHED_VARIABLE_SRC"] = df["MATCHED_VARIABLE_SRC"].str.replace(r"[;'\\\']", r"")
        df["OS"], df["BROWSER"] = zip(*list(df.CLIENT_USERAGENT.apply(parse_ua).values))

        df = self.__get_matched_variable_onehot(df)
        df = self.__get_browsers_onehot(df)

        os_low_freq = (df["OS"].value_counts()) / df.shape[0]
        os_low_freq = os_low_freq[os_low_freq <= 0.01]
        df.loc[df["OS"].isin(os_low_freq.index.tolist()), "OS"] = "Other"

        df = pd.get_dummies(df, columns=["OS"], dummy_na=True)
        df = df.drop(
            [
                "CLIENT_USERAGENT",
                "MATCHED_VARIABLE_NAME",
                "MATCHED_VARIABLE_VALUE",
                "MATCHED_VARIABLE_SRC",
                "BROWSER",
                "RESPONSE_CODE",
            ],
            axis=1,
        )

        ip2int = lambda ip: reduce(lambda a, b: int(a) * 256 + int(b), ip.split("."))
        df["CLIENT_IP"] = df["CLIENT_IP"].apply(ip2int)
        df["REQUEST_SIZE"] = df["REQUEST_SIZE"].astype(int)

        scaler = joblib.load("../models/scaler.pt")
        for feature in scaler.features:
            if feature not in df.columns:
                df[feature] = 0
        event = df["EVENT_ID"]
        df = pd.DataFrame(
            scaler.transform(df[scaler.features]),
            columns=df[scaler.features].columns,
        )
        df["EVENT_ID"] = event
        return df
