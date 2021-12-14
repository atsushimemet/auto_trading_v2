import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class Model:
    def __init__(self, datamart: pd.DataFrame):
        self._datamart = datamart

    @property
    def _target_col(self):
        return "target"

    @property
    def _single_value(self):
        return (
            self._datamart.drop(self._target_col, axis=1)
            .columns.tolist()[0]
            .split("_")[0]
        )

    @property
    def _num_lag(self):
        num_col = len(self._datamart.columns)
        num_col_target = 2  # NOTE: target & <single_value>_N-0
        return num_col - num_col_target

    @property
    def y(self):
        return self._datamart[self._target_col]

    @property
    def X(self):
        return self._datamart.drop(
            [self._target_col, self._single_value + "_N-0"], axis=1
        )

    def fit(self):
        self.clf = lgb.LGBMClassifier()
        self.clf.fit(self.X, self.y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict(X)

    def mse(self, true: pd.Series, pred: np.ndarray) -> float:
        return mean_squared_error(true, pred)
