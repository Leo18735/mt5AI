try:
    from MetaTrader5 import *
except ModuleNotFoundError:
    print("Using else")
    import pickle


    TIMEFRAME_D1 = 100


    def initialize():
        pass


    class SymbolInfo:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.point = 0.00001


    def symbol_info(symbol: str) -> SymbolInfo:
        return SymbolInfo(symbol)


    def copy_rates_from(*_, **__):
        with open("rates.pickle", "rb") as f:
            return pickle.load(f)
