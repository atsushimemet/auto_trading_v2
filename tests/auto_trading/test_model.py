import pandas as pd

from auto_trading.datamart import Datamart
from auto_trading.model import Model
from auto_trading.raw_data import RawData
from auto_trading.symbol_data import SymbolData


def create_datamart_msft() -> pd.DataFrame:
    symbol_data = SymbolData("MSFT").symbol_data
    raw_data = RawData(symbol_data).raw_data
    datamart = Datamart(raw_data, "close", 5)
    return datamart.datamart


def test_existing_model_class():
    datamart = create_datamart_msft()
    model = Model(datamart)
    assert model


def test_has_attr_datamart():
    datamart = create_datamart_msft()
    model = Model(datamart)
    assert len(model._datamart)


def test_lgb_clf_created():
    datamart = create_datamart_msft()
    model = Model(datamart)
    model.fit()
    assert model.clf
