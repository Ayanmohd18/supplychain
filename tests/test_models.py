import pytest
import pandas as pd
import numpy as np
import sys
import os
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.baseline import calculate_metrics
from pytorch_forecasting import TimeSeriesDataSet

def test_calculate_metrics():
    """Test the WMAPE calculation metric."""
    y_true = np.array([10, 20, 30])
    y_pred = np.array([9, 22, 30])
    
    mae, rmse, wmape = calculate_metrics(y_true, y_pred)
    
    assert mae == pytest.approx(1.0)
    assert wmape == pytest.approx(3.0 / 60.0)

def test_arima_baseline():
    """Test ARIMA baseline prediction on simple data."""
    # Create simple time series
    sales = np.random.poisson(lam=10, size=50).astype(float)
    
    horizon = 7
    train = sales[:-horizon]
    test = sales[-horizon:]
    
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=horizon)
    
    assert len(preds) == horizon
    assert not np.isnan(preds).any()

def test_tft_dataset_construction():
    """Test TFT TimeSeriesDataSet construction."""
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({
        'item_id': ['item_1'] * 100,
        'store_id': ['store_1'] * 100,
        'sales': np.random.poisson(lam=10, size=100).astype(float),
        'time_idx': np.arange(100),
        'day_of_week': dates.dayofweek.astype(str),
        'sell_price': [1.99] * 100
    })
    
    max_prediction_length = 7
    max_encoder_length = 14
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["item_id", "store_id"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["item_id", "store_id"],
        time_varying_known_categoricals=["day_of_week"],
        time_varying_known_reals=["time_idx", "sell_price"],
        time_varying_unknown_reals=["sales"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    assert isinstance(training, TimeSeriesDataSet)
    dataloader = training.to_dataloader(train=True, batch_size=4, num_workers=0)
    
    batch = next(iter(dataloader))
    x, y = batch
    
    assert 'encoder_cont' in x
    assert 'encoder_cat' in x
