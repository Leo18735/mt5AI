import numpy as np
import xgboost
from Classes.Models.Model import Model
import pandas as pd
import cupy as cp


class XGBoostModel(Model):
    def _prepare_train(self, x_train: np.ndarray, y_train: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros(shape=(x_train.shape[0], window * x_train.shape[1]))
        result[:] = np.nan

        for i in range(window-1, x_train.shape[0]):
            x = x_train[i-window+1:i+1]
            result[i] = x.reshape(np.prod(x.shape))

        mask = np.isnan(result)
        for i in reversed(range(len(result.shape) - 1)):
            mask = mask.any(axis=i + 1)

        self._model_shape = result.shape[1:]

        return result[~mask], y_train[~mask]

    def get_model(self):
        return xgboost.XGBClassifier(objective='binary:logistic',
                                     max_depth=10,
                                     learning_rate=0.3,
                                     n_estimators=500,
                                     random_state=1,
                                     device="gpu")

    def prepare_predict(self, x_test: pd.DataFrame, window: int) -> np.ndarray:
        return np.array(x_test.iloc[-window:])

    def predict(self, x_test: np.ndarray) -> int:
        return self._model.predict(self._scaler.transform(x_test).reshape(1, np.prod(x_test.shape)))

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self._model.fit(cp.array(x_train[:-1]), cp.array(y_train[1:]))
