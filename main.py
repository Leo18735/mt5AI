import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
from backtesting import Backtest  # ide is lying. backtesting is installed!!
from Classes.Dumper import Dumper
from Classes.MT5 import MT5
from Classes.MLStrategy import MLStrategy
import Classes.MetaTrader5 as MetaTrader5
from Classes.Models.XGBoostModel import XGBoostModel
import tqdm
from utils import create_variations, plot



def main():
    symbol: str = "EURUSD"
    timeframe: int = MetaTrader5.TIMEFRAME_D1
    amount: int = 12_410
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

    variations = create_variations({
        "window": range(10, 40, 5),
        "direction": range(100, 2_000, 50),
        "volume": [5000]
    }, lambda x: True)
    data = {"symbol": symbol, "timeframe": timeframe, "amount": amount, "split": test_split}

    dumper: Dumper = Dumper("dump.pickle")

    for variation in tqdm.tqdm(variations):
        key = {**data, "variation": variation}
        if dumper.exists(key):
            time.sleep(0.15)
            continue
        dumper.add((key, bt.run(**variation)))

    sort: str = "Win Rate [%]"

    sorted_result = sorted(dumper.get_results(), key=lambda x: x[1][sort], reverse=True)

    for result in sorted_result:
        print(f"{result[1][sort]}: {result[0]['variation']}")

    print("\n")
    print(bt.run(**sorted_result[0][0]["variation"]))
    bt.plot()
    plot(bt._data, bt._strategy.model.y)


if __name__ == '__main__':
    main()
