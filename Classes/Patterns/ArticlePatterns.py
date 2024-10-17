import pandas as pd

from Classes.Patterns.Patterns import Patterns


class ArticlePatterns(Patterns):
    @staticmethod
    def get_pattern(rates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        rates["Price_Change"] = rates["Close"].pct_change() * 100
        rates["raw_Std_Dev_Close"] = rates["Close"].rolling(window=20).std()

        rates["raw_Prev_Day_Price_Change"] = rates["Close"] - rates["Close"].shift(1)
        rates["raw_Prev_Week_Price_Change"] = rates["Close"] - rates["Close"].shift(7)
        rates["raw_Prev_Month_Price_Change"] = rates["Close"] - rates["Close"].shift(30)

        rates["Consecutive_Positive_Changes"] = (rates["Price_Change"] > 0).astype(int).groupby(
            (rates["Price_Change"] > 0).astype(int).diff().ne(0).cumsum()).cumsum()
        rates["Consecutive_Negative_Changes"] = (rates["Price_Change"] < 0).astype(int).groupby(
            (rates["Price_Change"] < 0).astype(int).diff().ne(0).cumsum()).cumsum()
        rates["Price_Density"] = rates["Close"].rolling(window=10).apply(lambda x: len(set(x)))
        rates["Fractal_Analysis"] = rates["Close"].rolling(window=10).apply(
            lambda x: 1 if x.idxmax() else (-1 if x.idxmin() else 0))
        rates["Median_Close_7"] = rates["Close"].rolling(window=7).median()
        rates["Median_Close_30"] = rates["Close"].rolling(window=30).median()
        rates["Price_Volatility"] = rates["Close"].rolling(window=20).std() / rates["Close"].rolling(
            window=20).mean()

        names = ["Price_Change", "raw_Std_Dev_Close",
                 "raw_Prev_Day_Price_Change", "raw_Prev_Week_Price_Change", "raw_Prev_Month_Price_Change",
                 "Consecutive_Positive_Changes", "Consecutive_Negative_Changes", "Price_Density",
                 "Fractal_Analysis", "Median_Close_7", "Median_Close_30", "Price_Volatility"]

        return rates, names
