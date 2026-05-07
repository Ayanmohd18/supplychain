import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ARIMABaseline:
    """
    Fits ARIMA or SARIMA models per item.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}

    def fit(self, df_train, id_col='id', time_col='d', target_col='sales'):
        # Train individually per item
        for item_id, group in df_train.groupby(id_col):
            y = group.sort_values(time_col)[target_col].values
            try:
                if sum(self.seasonal_order) > 0:
                    model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
                else:
                    model = ARIMA(y, order=self.order)
                self.models[item_id] = model.fit(disp=False)
            except Exception as e:
                # Fallback to simple mean if ARIMA fails to converge
                self.models[item_id] = np.mean(y)

    def predict(self, horizons=[1, 7, 14, 28]):
        predictions = []
        for item_id, model in self.models.items():
            max_h = max(horizons)
            if hasattr(model, 'forecast'):
                preds = model.forecast(steps=max_h)
            else:
                # Fallback mean
                preds = np.full(max_h, model)
            
            for h in horizons:
                if h <= len(preds):
                    predictions.append({'id': item_id, 'horizon': h, 'arima_forecast': preds[h-1]})
        return pd.DataFrame(predictions)


class NaiveBaselines:
    """
    Fits last value, seasonal naive, and moving average baselines.
    """
    def __init__(self, seasonal_period=7, ma_window=28):
        self.seasonal_period = seasonal_period
        self.ma_window = ma_window
        self.last_values = {}
        self.seasonal_values = {}
        self.ma_values = {}
        
    def fit(self, df_train, id_col='id', time_col='d', target_col='sales'):
        for item_id, group in df_train.groupby(id_col):
            y = group.sort_values(time_col)[target_col].values
            if len(y) > 0:
                self.last_values[item_id] = y[-1]
                self.seasonal_values[item_id] = y[-self.seasonal_period:] if len(y) >= self.seasonal_period else [y[-1]] * self.seasonal_period
                self.ma_values[item_id] = np.mean(y[-self.ma_window:]) if len(y) >= self.ma_window else np.mean(y)

    def predict(self, horizons=[1, 7, 14, 28]):
        preds = []
        for item_id in self.last_values.keys():
            last_v = self.last_values[item_id]
            ma_v = self.ma_values[item_id]
            season_v = self.seasonal_values[item_id]
            
            for h in horizons:
                preds.append({
                    'id': item_id, 
                    'horizon': h, 
                    'last_value': last_v,
                    'moving_average': ma_v,
                    'seasonal_naive': season_v[(h - 1) % len(season_v)]
                })
        return pd.DataFrame(preds)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # WMAPE: sum(|true - pred|) / sum(|true|)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1e-5)
    return mae, rmse, wmape
