import time
from backtesting import Backtest  # ide is lying. backtesting is installed!!
from Classes.Dumper import Dumper
from Classes.MT5 import MT5
from Classes.MLStrategy import MLStrategy
import MetaTrader5
from Classes.Models.XGBoostModel import XGBoostModel
import tqdm
from utils import create_variations, plot



def main():
    symbol: str = "EURUSD"
    timeframe: int = MetaTrader5.TIMEFRAME_D1
    amount: int = 15_000
    test_split: float = .15
    dump: bool = False

    mt5 = MT5(symbol)
    x_train, x_test = mt5.prepare_data(timeframe, amount, test_split, verbose=True)
    print(f"Train-Amount: {x_train.shape[0]}\n"
          f"Test-Amount: {x_test.shape[0]}\n"
          f"Signals: {x_train.shape[1]}")
    model = XGBoostModel(x_train, mt5)
    MLStrategy.model = model

    bt = Backtest(
        x_test,
        MLStrategy,
        **mt5.get_broker_conditions())

    variations = create_variations({
        "window": [25],
        "direction": [250],
        "volume": [10_000]
    }, lambda x: True)
    data = {"symbol": symbol, "timeframe": timeframe, "amount": amount, "split": test_split}

    dumper: Dumper = Dumper("dump.pickle", dump)

    for variation in tqdm.tqdm(variations):
        key = {**data, "variation": variation}
        if dumper.exists(key):
            time.sleep(0.15)
            continue
        dumper.add((key, bt.run(**variation)))

    sort: str = "Equity Final [$]"
    show: list[str] = ["# Trades"]

    sorted_result = sorted(dumper.get_results(), key=lambda x: x[1][sort], reverse=True)

    for result in sorted_result:
        show_attrs = {x: result[1][x] for x in show}
        print(f"{result[1][sort]}: {show_attrs}: {result[0]['variation']}")

    return
    print("\n")
    print(bt.run(**sorted_result[0][0]["variation"]))
    bt.plot()
    plot(bt._data, bt._strategy.model.y)


if __name__ == '__main__':
    main()
