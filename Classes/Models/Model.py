import abc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Classes.MT5 import MT5
import numpy as np


class Model(abc.ABC):
    def __init__(self, x_train: pd.DataFrame, mt5: MT5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train: pd.DataFrame = x_train
        self.mt5: MT5 = mt5
        self._scaler = None
        self._model = None

    def train(self, window: int, tp: int, sl: int):
        y_train = self.mt5.get_y(self.x_train, tp * self.mt5.points, sl * self.mt5.points)
        self._scaler = self.get_scaler()
        x_train: np.ndarray = self._scaler.fit_transform(np.array(self.x_train[self.mt5.signals]))
        x_train, y_train = self._prepare_train(x_train, y_train, window)
        self._model = self.get_model()
        self._fit(x_train[:-1], y_train[1:])

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self._model.fit(x_train[:-1], y_train[1:])

    @abc.abstractmethod
    def _prepare_train(self, x_train: np.ndarray, y_train: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def get_scaler():
        return StandardScaler()

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def prepare_predict(self, x_test: pd.DataFrame, window: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(self, x_test: np.ndarray) -> int:
        pass
