import pandas as pd
import datetime
import numpy as np
import pandas_ta as pd_ta
import Classes.MetaTrader5 as MetaTrader5


MetaTrader5.initialize()


class MT5:
    def __init__(self, symbol: str):
        self._symbol: str = symbol
        self.points = MetaTrader5.symbol_info(self._symbol).point
        self.signals: list[str] = []

    def prepare_data(self, timeframe: int, amount: int, test_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        rates: pd.DataFrame = self._get_rates(timeframe, amount)
        rates.Volume = 0

        rates["diff_open_close"] = rates["Close"] - rates["Open"]
        rates["diff_high_low"] = rates["High"] - rates["Low"]
        rates["rsi_7"] = pd_ta.rsi(rates["Close"], length=7)
        rates["rsi_14"] = pd_ta.rsi(rates["Close"], length=14)
        rates["ema_10"] = pd_ta.ema(rates["Close"], length=10) - rates["Close"]
        rates["ema_20"] = pd_ta.ema(rates["Close"], length=20) - rates["Close"]
        self.signals += ["diff_open_close", "diff_high_low",
                          "rsi_7", "rsi_14",
                          "ema_10", "ema_20"]

        bb_length = 5
        bb_std = 2.0
        bb = pd_ta.bbands(rates["Close"], length=bb_length, std=bb_std)
        for name in [f"BBL_{bb_length}_{bb_std}",
                    f"BBM_{bb_length}_{bb_std}",
                    f"BBU_{bb_length}_{bb_std}"]:
            rates[name] = bb[name] - rates["Close"]
            self.signals += [name]

        stoch_k = 14
        stoch_d = 3
        stoch_smooth_k = 3
        stoch = pd_ta.stoch(rates["High"], rates["Low"], rates["Close"], k=stoch_k, d=stoch_d, smooth_k=stoch_smooth_k)
        for name in [f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}",
                     f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"]:
            rates[name] = stoch[name]
            self.signals += [name]

        adx_length = 14
        adx_lensig = adx_length
        adx = pd_ta.adx(rates["High"], rates["Low"], rates["Close"], length=adx_length, adx_lensig=adx_lensig, scalar=100, mamode="rma")
        for name in [f"ADX_{adx_lensig}",
                     f"DMP_{adx_length}",
                     f"DMN_{adx_length}"]:
            rates[name] = adx[name]
            self.signals += [name]

        split_index: int = int(rates.shape[0] * (1 - test_split))
        return rates.iloc[:split_index], rates.iloc[split_index:]

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
    def get_y(x_train: pd.DataFrame, tp: float, sl: float):
        def check_buy(cut_rates: pd.DataFrame) -> bool:
            price = cut_rates["Open"].iloc[0]
            for f in range(cut_rates.shape[0]):
                if cut_rates["Low"].iloc[f] < price - sl:
                    return False
                if cut_rates["High"].iloc[f] > price + tp:
                    return True
            return False

        def check_sell(cut_rates: pd.DataFrame) -> bool:
            price = cut_rates["Open"].iloc[0]
            for f in range(cut_rates.shape[0]):
                if cut_rates["High"].iloc[f] > price + sl:
                    return False
                if cut_rates["Low"].iloc[f] < price - tp:
                    return True
            return False

        y = pd.Series(1, index=x_train["Close"].index)
        for i in range(0, x_train.shape[0]):
            buy = check_buy(x_train.iloc[i:])
            sell = check_sell(x_train.iloc[i:])
            if buy and sell:
                continue
            if buy:
                y.iloc[i] = 2
            if sell:
                y.iloc[i] = 0

        return y

