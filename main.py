from backtesting import Backtest
from MT5 import MT5
from Model import TFModel, XGBoost
from MLStrategy import MLStrategy
import MetaTrader5



def prep_env(model_object, volume, epochs, batch_size, rates_amount, timeframe, test_size, symbol, window, threshold, load):
    mt5 = MT5(symbol, window)
    x_train, x_test, y_train = mt5.get_data(rates_amount, test_size, timeframe, threshold * mt5.get_points())
    model_object.set_trained_model(x_train, y_train, mt5, epochs, batch_size, load)
    MLStrategy.set_values(model_object, volume, window, mt5, x_test.shape[0])
    return x_test


def main(model_class, volume, epochs, batch_size, rates_amount, timeframe, test_size, symbol, window, cash, commission, margin, threshold, load):
    x_test = prep_env(model_class, volume, epochs, batch_size, rates_amount, timeframe, test_size, symbol, window, threshold, load)

    bt = Backtest(
        x_test,
        MLStrategy,
        cash=cash,
        commission=commission,
        margin=margin,
        trade_on_close=True)

    result = bt.run()
    print(result)
    bt.plot()


def run():
    volume = 5_000
    cash = 10_000
    commission = .0006
    margin = 1 / 30

    epochs = 100
    batch_size = 32

    window = 30
    threshold = 30

    rates_amount = 25_000
    timeframe = MetaTrader5.TIMEFRAME_D1
    test_size = .1
    symbol = "EURUSD"

    load = True
    model_class = XGBoost

    main(model_class(), volume, epochs, batch_size, rates_amount, timeframe, test_size, symbol, window, cash, commission, margin, threshold, load)


if __name__ == '__main__':
    run()
