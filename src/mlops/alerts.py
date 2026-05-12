import pandas as pd
import numpy as np

class SupplyChainAlerts:
    """
    Analyzes forecasts to identify high-risk events (stockouts, overstock, drift).
    """
    def __init__(self, threshold_stockout=0.2, threshold_overstock=0.8):
        self.threshold_stockout = threshold_stockout
        self.threshold_overstock = threshold_overstock

    def detect_risks(self, item_id, forecast_p10, forecast_p90, current_inventory):
        """
        Detects if an item is at risk of stockout or overstock.
        """
        alerts = []
        
        # Predicted demand for next 7 days (mean of P10 to P90)
        expected_demand = (forecast_p10 + forecast_p90) / 2
        total_expected_demand = np.sum(expected_demand)
        
        # 1. Stockout Risk: Current Inventory < Expected Demand (Lower Bound)
        if current_inventory < np.sum(forecast_p10):
            alerts.append({
                "type": "CRITICAL",
                "item": item_id,
                "message": f"High Stockout Risk: Inventory ({current_inventory}) below P10 demand ({np.sum(forecast_p10):.0f}).",
                "severity": "high"
            })
        elif current_inventory < total_expected_demand:
            alerts.append({
                "type": "WARNING",
                "item": item_id,
                "message": "Potential Stockout: Inventory may not meet median expected demand.",
                "severity": "medium"
            })
            
        # 2. Overstock Risk: Inventory > 2x P90 demand
        if current_inventory > (2 * np.sum(forecast_p90)):
            alerts.append({
                "type": "EFFICIENCY",
                "item": item_id,
                "message": "Excessive Overstock: Capital tied up in slow-moving inventory.",
                "severity": "low"
            })
            
        return alerts

def get_mock_alerts():
    """Generates a list of mock alerts for demonstration."""
    return [
        {"type": "CRITICAL", "item": "FOODS_3_090_CA_1", "message": "Lead time delay + High demand spike predicted. ETA Stockout: 3 days.", "severity": "high"},
        {"type": "WARNING", "item": "HOBBIES_1_001_CA_1", "message": "Price volatility increasing. Forecast variance exceeds tolerance.", "severity": "medium"},
        {"type": "STABILITY", "item": "HOUSEHOLD_2_001_CA_1", "message": "Model drift detected in this category (PSI: 0.18).", "severity": "low"}
    ]
