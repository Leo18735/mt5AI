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
        self.y: pd.Series = None
        self._scaler = None
        self._model = None
        self._model_shape = None

    def train(self, window: int, direction: int):
        y_train = self.mt5.get_y(self.x_train, direction * self.mt5.points)
        self._scaler = self.get_scaler()
        x_train: np.ndarray = self._scaler.fit_transform(np.array(self.x_train[self.mt5.signals]))
        x_train = self._prepare_train(x_train, window)
        x_train, y_train = self._apply_mask(x_train, y_train)
        self._model = self.get_model()
        self._fit(x_train[:-1], y_train[1:])

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self._model.fit(x_train[:-1], y_train[1:])

    @abc.abstractmethod
    def _prepare_train(self, x_train: np.ndarray, window: int) -> np.ndarray:
        pass

    @staticmethod
    def get_scaler():
        return StandardScaler()

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def predict(self, x_test: pd.DataFrame, window: int):
        pass

    def _apply_mask(self, x_train: np.ndarray, y_train: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isnan(x_train)
        for i in reversed(range(len(x_train.shape) - 1)):
            mask = mask.any(axis=i + 1)

        self._model_shape = x_train.shape[1:]

        if y_train is not None:
            return x_train[~mask], y_train[~mask]
        return x_train[~mask]
