import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import M5Loader
from src.data.feature_engineering import create_base_features

@pytest.fixture
def loader():
    return M5Loader(data_dir='../data')

def test_loader_load_all(loader):
    """Test loading of raw datasets."""
    calendar, sales, prices = loader.load_all()
    
    assert isinstance(calendar, pd.DataFrame)
    assert isinstance(sales, pd.DataFrame)
    assert isinstance(prices, pd.DataFrame)
    
    assert not calendar.empty
    assert not sales.empty
    assert not prices.empty

def test_loader_merged_sample(loader):
    """Test loading and merging a subset of data."""
    df = loader.load_merged_sample(n_items=5)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    # Assert expected columns are present from the merge
    expected_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'sales', 'date', 'wm_yr_wk']
    for col in expected_cols:
        assert col in df.columns

def test_feature_engineering():
    """Test feature engineering logic."""
    # Create mock dataframe matching expected structure
    dates = pd.date_range(start='2011-01-29', periods=10, freq='D')
    df = pd.DataFrame({
        'id': ['item_1'] * 10,
        'item_id': ['item_1_id'] * 10,
        'sales': [1, 2, 0, 4, 5, 2, 1, 0, 3, 4],
        'date': dates,
        'sell_price': [1.99] * 10
    })
    
    # Run feature engineering
    df_features = create_base_features(df)
    
    # Check that new features are added
    assert 'day_of_week' in df_features.columns
    assert 'month' in df_features.columns
    assert 'year' in df_features.columns
    
    # Check for absence of NaNs in key required columns
    assert not df_features['day_of_week'].isna().any()
    assert not df_features['month'].isna().any()
