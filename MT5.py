import datetime
import pandas as pd
import MetaTrader5
import numpy as np
import pandas_ta as pd_ta
from sklearn.model_selection import train_test_split
import pickle

MetaTrader5.initialize()


class MT5:
    def __init__(self, symbol, window):
        self._symbol = symbol
        self._window = window
        self._signals = []
        self._points = MetaTrader5.symbol_info(self._symbol).point

    def get_rates(self, amount: int, timeframe):
        rates = MetaTrader5.copy_rates_from(
            self._symbol,
            timeframe,
            datetime.datetime(year=2024, month=1, day=1),
            amount
        )
        with open("rates.pickle", "wb") as f:
            pickle.dump(rates, f)
        return pd.DataFrame(
            rates,
            index=np.array([datetime.datetime.fromtimestamp(x) for x in rates["time"]])
        )[["open", "high", "low", "close", "real_volume"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "real_volume": "Volume"})

    def get_points(self):
        return self._points

    def get_signals(self):
        return self._signals

    def prepare_window(self, x_train: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.zeros(shape=(x_train.shape[0], self._window, x_train.shape[1]))
        result[:] = np.nan

        for i in range(self._window, x_train.shape[0]):
            result[i] = x_train[i - self._window:i]

        return result

    @staticmethod
    def remove_nan(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isnan(x_train).any(axis=2).any(axis=1)
        return x_train[~mask], y_train[~mask]

    def get_data(self, amount: int, test_size: float, timeframe: int, threshold: float) \
            -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        rates = self.get_rates(amount, timeframe)
        rates.Volume = 0

        rates["diff_open_close"] = rates["Close"] - rates["Open"]
        rates["diff_high_low"] = rates["High"] - rates["Low"]
        rates["rsi_7"] = pd_ta.rsi(rates["Close"], length=7)
        rates["rsi_14"] = pd_ta.rsi(rates["Close"], length=14)
        rates["ema_10"] = pd_ta.ema(rates["Close"], length=10) - rates["Close"]
        rates["ema_20"] = pd_ta.ema(rates["Close"], length=20) - rates["Close"]
        self._signals += ["diff_open_close", "diff_high_low",
                          "rsi_7", "rsi_14",
                          "ema_10", "ema_20"]

        bb_length = 5
        bb_std = 2.0
        bb = pd_ta.bbands(rates["Close"], length=bb_length, std=bb_std)
        for name in [f"BBL_{bb_length}_{bb_std}",
                    f"BBM_{bb_length}_{bb_std}",
                    f"BBU_{bb_length}_{bb_std}"]:
            rates[name] = bb[name] - rates["Close"]
            self._signals += [name]

        stoch_k = 14
        stoch_d = 3
        stoch_smooth_k = 3
        stoch = pd_ta.stoch(rates["High"], rates["Low"], rates["Close"], k=stoch_k, d=stoch_d, smooth_k=stoch_smooth_k)
        for name in [f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}",
                     f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"]:
            rates[name] = stoch[name]
            self._signals += [name]

        adx_length = 14
        adx_lensig = adx_length
        adx = pd_ta.adx(rates["High"], rates["Low"], rates["Close"], length=adx_length, adx_lensig=adx_lensig, scalar=100, mamode="rma")
        for name in [f"ADX_{adx_lensig}",
                     f"DMP_{adx_length}",
                     f"DMN_{adx_length}"]:
            rates[name] = adx[name]
            self._signals += [name]

        y = pd.Series(1, index=rates["Close"].index)
        y[rates["Close"] - rates["Open"]> threshold] = 2
        y[rates["Open"] - rates["Close"] > threshold] = 0

        x_train, x_test, y_train, _ = train_test_split(
            rates[:-1],
            y[1:],
            test_size=test_size,
            shuffle=False)

        return np.array(x_train[self.get_signals()]), x_test, np.array(y_train)
