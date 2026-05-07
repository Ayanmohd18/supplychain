import pandas as pd
import numpy as np

def create_base_features(df):
    """
    Standard feature engineering for M5 data.
    """
    # Date features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Lag features (example)
    # df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
    
    return df
