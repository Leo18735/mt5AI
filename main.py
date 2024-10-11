from backtesting import Backtest
from Classes.MT5 import MT5
from Classes.MLStrategy import MLStrategy
import Classes.MetaTrader5 as MetaTrader5
from Classes.Models.TFModel import TFModel
from Classes.Models.XGBoostModel import XGBoostModel


def main():
    symbol: str = "EURUSD"
    timeframe: int = MetaTrader5.TIMEFRAME_D1
    amount: int = 3_000
    test_split: float = .05
    MLStrategy.model = XGBoostModel()

    mt5 = MT5(symbol)
    x_train, x_test = mt5.prepare_data(timeframe, amount, test_split)
    MLStrategy.model.x_train = x_train
    MLStrategy.model.mt5 = mt5

    MLStrategy.volume = 5_000

    bt = Backtest(
        x_test,
        MLStrategy,
        **mt5.get_broker_conditions(),
        trade_on_close=True)
    print("Run")

    _, heatmap = bt.optimize(
        window=range(5, 45, 5),
        sl=range(100, 500, 50),
        tp=range(100, 500, 50),
        constraint=lambda x: x.window and x.tp and x.sl and x.tp >= x.sl,
        maximize='Equity Final [$]',
        return_heatmap=True
    )
    print(heatmap)
    bt.plot()


if __name__ == '__main__':
    main()
