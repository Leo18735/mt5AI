import pandas as pd
import pandas_ta as pd_ta
from Classes.Patterns.Patterns import Patterns


class CustomPatterns(Patterns):
    @staticmethod
    def get_pattern(rates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        names: list[str] = []

        rates["diff_open_close"] = rates["Close"] - rates["Open"]
        rates["diff_high_low"] = rates["High"] - rates["Low"]
        rates["rsi_7"] = pd_ta.rsi(rates["Close"], length=7)
        rates["rsi_14"] = pd_ta.rsi(rates["Close"], length=14)
        rates["ema_10"] = pd_ta.ema(rates["Close"], length=10) - rates["Close"]
        rates["ema_20"] = pd_ta.ema(rates["Close"], length=20) - rates["Close"]
        rates["atr"] = pd_ta.atr(rates["High"], rates["Low"], rates["Close"], length=14, mamode="rma")
        names += ["diff_open_close", "diff_high_low",
                         "rsi_7", "rsi_14",
                         "ema_10", "ema_20",
                         "atr"]

        af0 = 0.02
        max_af = 0.2
        psar = pd_ta.psar(rates["High"], rates["Low"], rates["Close"], af0=af0, max_af=max_af)
        psar[psar.isna()] = 0
        for name in [f"PSARl_{af0}_{max_af}",
                     f"PSARs_{af0}_{max_af}",
                     f"PSARaf_{af0}_{max_af}",
                     f"PSARr_{af0}_{max_af}"]:
            rates[name] = psar[name] - rates["Close"]
            names += [name]

        bb_length = 5
        bb_std = 2.0
        bb = pd_ta.bbands(rates["Close"], length=bb_length, std=bb_std)
        for name in [f"BBL_{bb_length}_{bb_std}",
                    f"BBM_{bb_length}_{bb_std}",
                    f"BBU_{bb_length}_{bb_std}"]:
            rates[name] = bb[name] - rates["Close"]
            names += [name]

        tenkan = 9
        kijun = 26
        senkou = 52
        ichimoku = pd_ta.ichimoku(rates["High"], rates["Low"], rates["Close"], tenkan, kijun, senkou)[0]
        for name in [f"ISA_{tenkan}",
                     f"ISB_{kijun}",
                     f"ITS_{tenkan}",
                     f"IKS_{kijun}",
                     ]:  # f"ICS_{kijun}"]:  # this is doing a forecast!!!
            rates[name] = ichimoku[name] - rates["Close"]
            names += [name]

        stoch_k = 14
        stoch_d = 3
        stoch_smooth_k = 3
        stoch = pd_ta.stoch(rates["High"], rates["Low"], rates["Close"], k=stoch_k, d=stoch_d, smooth_k=stoch_smooth_k)
        for name in [f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}",
                     f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"]:
            rates[name] = stoch[name]
            names += [name]

        adx_length = 14
        adx_lensig = adx_length
        adx = pd_ta.adx(rates["High"], rates["Low"], rates["Close"], length=adx_length, adx_lensig=adx_lensig, scalar=100, mamode="rma")
        for name in [f"ADX_{adx_lensig}",
                     f"DMP_{adx_length}",
                     f"DMN_{adx_length}"]:
            rates[name] = adx[name]
            names += [name]

        return rates, names
