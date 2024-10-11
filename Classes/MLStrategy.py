from backtesting import Strategy  # ide is lying. backtesting is installed!!
from Classes.Models.Model import Model
import tqdm


class MLStrategy(Strategy):
    model: Model = None
    window: int = None
    tp: int = None
    sl: int = None
    volume = None

    _tqdm = None
    tqdm = False

    def init(self):
        self.model.train(self.window, self.tp, self.sl)
        if self.tqdm:
            self._tqdm = tqdm.tqdm(total=self.data.df.shape[0], leave=False)

    def next(self):
        if self.tqdm:
            self._tqdm.update(1)
        x_test = self.model.prepare_predict(self.data.df[self.model.mt5.signals], self.window)
        if x_test.shape[0] < self.window:
            return
        prediction = self.model.predict(x_test)
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
