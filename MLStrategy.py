from backtesting import Strategy
import numpy as np
from tqdm import tqdm


class MLStrategy(Strategy):
    model_object = None
    volume = None
    window = None
    scaler_x = None
    model = None
    mt5 = None
    points = None
    x_size = None

    tqdm = None

    @classmethod
    def set_values(cls, model_object, volume: float, window: int, mt5, x_size: int):
        cls.model_object = model_object
        cls.volume = volume
        cls.window = window
        cls.mt5 = mt5
        cls.points = mt5.get_points()
        cls.x_size = x_size

    def init(self):
        self.tqdm = tqdm(initial=0, total=self.x_size, leave=False)

    def next(self):
        self.tqdm.update(1)
        x_test = np.array(self.data.df[self.mt5.get_signals()])[-self.window:]
        if x_test.shape[0] < self.window:
            return
        prediction = self.model_object.predict(x_test)
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
