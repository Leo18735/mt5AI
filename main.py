import time

from backtesting import Backtest
from Classes.Dumper import Dumper
from Classes.MT5 import MT5
from Classes.MLStrategy import MLStrategy
import Classes.MetaTrader5 as MetaTrader5
from Classes.Models.TFModel import TFModel
from Classes.Models.XGBoostModel import XGBoostModel
import tqdm


def create_variations(args: dict) -> list[dict]:
    variations = [{}]
    for key, values in args.items():
        variations_old = variations.copy()
        variations = []
        for value in values:
            for variation in variations_old:
                variations.append({**variation, key: value})
    return variations


def main():
    symbol: str = "EURUSD"
    timeframe: int = MetaTrader5.TIMEFRAME_D1
    amount: int = 15_000
    test_split: float = .1

    mt5 = MT5(symbol)
    x_train, x_test = mt5.prepare_data(timeframe, amount, test_split)
    model = XGBoostModel(x_train, mt5)
    MLStrategy.model = model

    bt = Backtest(
        x_test,
        MLStrategy,
        **mt5.get_broker_conditions(),
        trade_on_close=True)
    print("Run")

    variations = create_variations({
        "window": range(5, 50, 5),
        "sl": range(100, 500, 50),
        "tp": range(100, 500, 50),
        "volume": [5000]
    })

    dumper: Dumper = Dumper("dump.pickle")

    for variation in tqdm.tqdm(variations):
        if dumper.exists(variation):
            time.sleep(.2)
            continue
        dumper.add((variation, bt.run(**variation)))

    for result in sorted(dumper.get_results(), key=lambda x: x[1]["Equity Final [$]"]):
        print(f"{result[1]['Equity Final [$]']}: {result[0]}")


if __name__ == '__main__':
    main()
