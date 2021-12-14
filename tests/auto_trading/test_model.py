import pandas as pd

from auto_trading.datamart import Datamart
from auto_trading.model import Model
from auto_trading.raw_data import RawData
from auto_trading.symbol_data import SymbolData


def create_datamart_msft() -> pd.DataFrame:
    symbol_data = SymbolData("MSFT").symbol_data
    raw_data = RawData(symbol_data).raw_data
    datamart = Datamart(raw_data, "close", 5)
    return datamart


def test_existing_model_class():
    datamart = create_datamart_msft()
    assert Model(datamart)


def test_has_attr_datamart():
    datamart = create_datamart_msft()
    model = Model(datamart)
    assert model._datamart
