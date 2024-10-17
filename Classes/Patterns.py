import pandas as pd
from PatternPy.tradingpatterns.tradingpatterns import (
    detect_head_shoulder,
    detect_multiple_tops_bottoms,
    calculate_support_resistance,
    detect_triangle_pattern,
    detect_wedge,
    detect_channel,
    detect_double_top_bottom,
    detect_trendline,
    find_pivots)

class Patterns:
    @staticmethod
    def apply_patterns(rates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        all_names: list[str] = []
        data, name = detect_head_shoulder(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, name = detect_multiple_tops_bottoms(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, names = calculate_support_resistance(rates.copy())
        all_names += names
        for name in names:
            data.loc[:, name] -= rates["Close"]
        rates.loc[:, names] = data[names]

        data, name = detect_triangle_pattern(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, name = detect_wedge(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, name = detect_channel(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, name = detect_double_top_bottom(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        data, names = detect_trendline(rates.copy())
        all_names += names
        for name in names:
            data.loc[:, name] -= rates["Close"]
        rates.loc[:, names] = data[names]

        data, name = find_pivots(rates.copy())
        all_names.append(name)
        rates.loc[:, name] = data[name]

        return rates, all_names
