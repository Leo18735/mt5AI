import time
import pandas as pd
import matplotlib.pyplot as plt


def create_variations(args: dict, condition) -> list[dict]:
    variations = [{}]
    for key, values in args.items():
        variations_old = variations.copy()
        variations = []
        for value in values:
            for variation in variations_old:
                variations.append({**variation, key: value})
    return [x for x in variations if condition(x)]


def time_it(func):
    def wrapper(*args, **kwargs):
        time0 = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - time0}")
        return result
    return wrapper


def plot(stock_prices: pd.DataFrame, y: pd.Series, width=.3):
    y = y.copy()
    y[y.isna()] = 1
    plt.figure()
    up = stock_prices[y == 2]
    color_up = "green"
    even = stock_prices[y == 1]
    color_even = "blue"
    down = stock_prices[y == 0]
    color_down = "red"

    plt.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color=color_up)
    plt.bar(up.index, up.High - up.Close, .1 * width, bottom=up.Close, color=color_up)
    plt.bar(up.index, up.Low - up.Open, .1 * width, bottom=up.Open, color=color_up)

    plt.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color=color_down)
    plt.bar(down.index, down.High - down.Open, .1 * width, bottom=down.Open, color=color_down)
    plt.bar(down.index, down.Low - down.Close, .1 * width, bottom=down.Close, color=color_down)

    plt.bar(even.index, even.Close - even.Open, width, bottom=even.Open, color=color_even)
    plt.bar(even.index, even.High - even.Open, .1 * width, bottom=even.Open, color=color_even)
    plt.bar(even.index, even.Low - even.Close, .1 * width, bottom=even.Close, color=color_even)

    plt.xticks(rotation=30, ha='right')
    plt.show()
