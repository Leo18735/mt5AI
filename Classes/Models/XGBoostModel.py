import numpy as np
import xgboost
from Classes.Models.Model import Model
import pandas as pd


class XGBoostModel(Model):
    def _prepare_train(self, x_train: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros(shape=(x_train.shape[0], window * x_train.shape[1]))
        result[:] = np.nan

        for i in range(window-1, x_train.shape[0]):
            x = x_train[i-window+1:i+1]
            result[i] = x.reshape(np.prod(x.shape))

        return result

    def get_model(self):
        return xgboost.XGBClassifier(objective='binary:logistic',
                                     max_depth=10,
                                     learning_rate=0.3,
                                     n_estimators=500,
                                     random_state=1,
                                     device="gpu")

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self._model.fit(x_train[:-1], y_train[1:])

    def predict(self, x_test: pd.DataFrame, window: int) -> np.ndarray:
        return self._model.predict(
            self._prepare_train(
                self._scaler.fit_transform(np.array(x_test[self.mt5.signals])),
                window
            )
        )
