import numpy as np
from backtesting import Strategy  # ide is lying. backtesting is installed!!
from Classes.Models.Model import Model


class MLStrategy(Strategy):
    model: Model = None
    window: int = None
    direction: int = None
    volume = None

    def init(self):
        self.model.train(self.window, self.direction)
        self.data.df["y"] = np.nan
        self.data.df.loc[self.data.df.index[self.window-1]:, "y"] = self.model.predict(self.data.df, self.window)[self.window - 1:]
        self.model.y = self.data.df.loc[:, "y"]

    def next(self):
        prediction = self.data.df["y"].iloc[-1]
        if prediction == 2:
            if not self.position:
                self.buy(size=self.volume)
            elif self.position.is_short:
                self.position.close()
                self.buy(size=self.volume)
        if prediction == 0:
            if not self.position:
                self.sell(size=self.volume)
            elif self.position.is_long:
                self.position.close()
                self.sell(size=self.volume)
