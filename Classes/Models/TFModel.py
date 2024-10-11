import numpy as np
import pandas as pd
from Classes.Models.Model import Model
import keras


class TFModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model_shape: tuple = None

    def _prepare_train(self, x_train: np.ndarray, y_train: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros(shape=(x_train.shape[0], window, x_train.shape[1]))
        result[:] = np.nan
        for i in range(window-1, x_train.shape[0]):
            result[i] = x_train[i-window+1:i+1]

        mask = np.isnan(result)
        for i in reversed(range(len(result.shape)-1)):
            mask = mask.any(axis=i+1)

        self._model_shape = result.shape[1:]

        return result[~mask], y_train[~mask]

    def prepare_predict(self, x_test: pd.DataFrame, window: int) -> np.ndarray:
        return np.array(x_test.iloc[-window:])

    def get_model(self):
        model = keras.models.Sequential([
            keras.layers.Input(self._model_shape),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.LSTM(50),
            keras.layers.Flatten(),
            keras.layers.Dense(50),
            keras.layers.Dense(50),
            keras.layers.Dense(50),
            keras.layers.Dense(3, activation="sigmoid")
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    def predict(self, x_test: np.ndarray) -> int:
        return np.argmax(self._model.predict(self._scaler.transform(x_test).reshape(1, *x_test.shape), verbose=0))
