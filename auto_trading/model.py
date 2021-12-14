import pandas as pd


class Model:
    def __init__(self, datamart: pd.DataFrame):
        self._datamart = datamart
