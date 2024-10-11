import os.path
import datetime
import subprocess
import abc
import xgboost
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras
from MT5 import MT5


class Model(abc.ABC):
    def __init__(self):
        self._model = None
        self._scaler_x = None

    @staticmethod
    @abc.abstractmethod
    def get_model(*_, **__):
        pass

    @abc.abstractmethod
    def set_trained_model(self, *_, **__):
        pass

    @abc.abstractmethod
    def predict(self, *_):
        pass


class TFModel(Model):
    @staticmethod
    def get_model(x_train: np.ndarray):
        model = keras.models.Sequential([
            keras.layers.Input((*x_train.shape[1:],)),
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

    def set_trained_model(self, x_train: np.ndarray, y_train: np.ndarray, mt5: MT5, epochs: int, batch_size: int, load):
        model_folder: str = "cache"
        model_file = os.path.join(model_folder, "model.keras")
        scaler_file = os.path.join(model_folder, "scaler_x")
        if os.path.exists(model_file) and os.path.exists(scaler_file) and load:
            self._model = keras.models.load_model(model_file)
            self._scaler_x = joblib.load(scaler_file)
            return
        self._scaler_x = StandardScaler()
        x_train = self._scaler_x.fit_transform(x_train)
        x_train = mt5.prepare_window(x_train)
        x_train, y_train = mt5.remove_nan(x_train, y_train)
        self._model = self.get_model(x_train)
        log_dir = os.path.join(model_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if load:
            p = subprocess.Popen(["tensorboard", "--logdir", model_folder],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("http://localhost:6006/")
        self._model.fit(x_train,
                          y_train.reshape(-1, 1),
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose="auto" if load else 0,
                          callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)] if load else None)
        if load:
            p.kill()
            self._model.save(model_file)
            joblib.dump(self._scaler_x, scaler_file)

    def predict(self, x_test: np.ndarray):
        return np.argmax(self._model.predict(self._scaler_x.transform(x_test).reshape(1, *x_test.shape), verbose=0))


class XGBoost(Model):
    @staticmethod
    def get_model(*_):
        return xgboost.XGBClassifier(objective='binary:logistic', max_depth=10, learning_rate=0.3,
                                n_estimators=500, random_state=1)

    def set_trained_model(self, x_train: np.ndarray, y_train: np.ndarray, mt5: MT5, *_, **__):
        self._model = self.get_model()
        self._scaler_x = StandardScaler()
        x_train = self._scaler_x.fit_transform(x_train)
        x_train = mt5.prepare_window(x_train)
        x_train, y_train = mt5.remove_nan(x_train, y_train)
        self._model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray):
        return 0
