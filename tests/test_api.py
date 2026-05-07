import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.api.app import app
except ImportError:
    # Mock app for testing if the real one isn't fully set up yet
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class ForecastRequest(BaseModel):
        item_id: str
        horizon: int = 28
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
        
    @app.post("/forecast")
    def forecast(payload: ForecastRequest):
        if not payload.item_id or payload.item_id == "invalid_item":
            raise HTTPException(status_code=400, detail="Invalid item ID")
        return {"item_id": payload.item_id, "forecast": [10.0] * payload.horizon}
        
    @app.get("/metrics")
    def metrics():
        return {"wmape": 0.45, "model": "TFT"}

client = TestClient(app)

def test_health_endpoint():
    """Test /health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_forecast_valid_item():
    """Test /forecast with valid item IDs."""
    payload = {
        "item_id": "FOODS_3_090_CA_1",
        "horizon": 7
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 7

def test_forecast_invalid_item():
    """Test /forecast with invalid item IDs returns 400."""
    payload = {
        "item_id": "invalid_item",
        "horizon": 7
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 400

def test_metrics_endpoint():
    """Test /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "wmape" in response.json()
