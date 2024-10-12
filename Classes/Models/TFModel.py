import numpy as np
import pandas as pd
from Classes.Models.Model import Model
import keras


class TFModel(Model):

    def _prepare_train(self, x_train: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros(shape=(x_train.shape[0], window, x_train.shape[1]))
        result[:] = np.nan
        for i in range(window-1, x_train.shape[0]):
            result[i] = x_train[i-window+1:i+1]

        self._model_shape = result.shape[1:]
        return result

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
