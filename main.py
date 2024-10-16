import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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
    timeframe: int = MetaTrader5.TIMEFRAME_H1
    amount: int = 20_000 * 24
    test_split: float = .01
    dump: bool = False

    mt5 = MT5(symbol)
    x_train, x_test = mt5.prepare_data(timeframe, amount, test_split)
    model = XGBoostModel(x_train, mt5)
    MLStrategy.model = model

    bt = Backtest(
        x_test,
        MLStrategy,
        **mt5.get_broker_conditions())

    variations = create_variations({
        "window": range(20, 35, 5),
        "direction": range(300, 450, 50),
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

    print("\n")
    print(bt.run(**sorted_result[0][0]["variation"]))
    bt.plot()
    plot(bt._data, bt._strategy.model.y)


if __name__ == '__main__':
    main()
