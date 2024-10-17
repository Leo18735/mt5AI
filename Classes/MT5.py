import pandas as pd
import datetime
import numpy as np
import MetaTrader5
from Classes.Patterns.ArticlePatterns import ArticlePatterns
from Classes.Patterns.CustomPatterns import CustomPatterns
from Classes.Patterns.PatternPyPatterns import PatternPyPatters

MetaTrader5.initialize()


class MT5:
    def __init__(self, symbol: str):
        self._symbol: str = symbol
        self.points = MetaTrader5.symbol_info(self._symbol).point
        self.signals: list[str] = []

    def prepare_data(self, timeframe: int, amount: int, test_split: float, verbose: bool = False) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        if verbose:
            print("Get rates")
        rates: pd.DataFrame = self._get_rates(timeframe, amount)
        rates.Volume = 0

        if verbose:
            print("Calculate Custom signals")
        rates, names = CustomPatterns.get_pattern(rates.copy())
        self.signals += names

        if verbose:
            print("Calculate PatternPy signals")
        rates, names = PatternPyPatters.get_pattern(rates.copy())
        self.signals += names
        
        if verbose:
            print("Calculate Article signals")
        rates, names = ArticlePatterns.get_pattern(rates.copy())
        self.signals += names

        if verbose:
            print("split")
        split_index: int = int(rates.shape[0] * (1 - test_split))
        result = rates.iloc[:split_index], rates.iloc[split_index:]
        print("Done")
        return result

    def _get_rates(self, timeframe: int, amount: int) -> pd.DataFrame:
        rates = MetaTrader5.copy_rates_from(
            self._symbol,
            timeframe,
            datetime.datetime(year=2024, month=1, day=1),
            amount
        )
        return pd.DataFrame(
            rates,
            index=np.array([datetime.datetime.fromtimestamp(x) for x in rates["time"]])
        )[["open", "high", "low", "close", "real_volume"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "real_volume": "Volume"})

    @staticmethod
    def get_broker_conditions() -> dict:
        return {"cash": 10_000,
                "commission": .0006,
                "margin": 1 / 30}

    @staticmethod
    def get_y(x_train: pd.DataFrame, direction: float):
        y = pd.Series(1, index=x_train["Close"].index)
        for i in range(0, x_train.shape[0]):
            price = x_train.loc[x_train.index[i], "Open"]
            buy = False
            sell = False
            for f in range(i, x_train.shape[0]):
                if x_train.loc[x_train.index[f], "High"] > price + direction:
                    buy = True
                if x_train.loc[x_train.index[f], "Low"] < price - direction:
                    sell = True
                if buy or sell:
                    break

            if buy and sell:
                continue
            if buy:
                y.iloc[i] = 2
            if sell:
                y.iloc[i] = 0

        return y

