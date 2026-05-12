import requests
import json

class DisruptionRadar:
    """
    SOTA Feature: External Signal Processing.
    Uses Ollama to turn unstructured news into a 'Demand Multiplier'.
    """
    def __init__(self, model="llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def analyze_external_signals(self, news_text):
        """
        Takes raw news/weather text and returns a disruption score.
        Score > 1.0 = Unexpected Demand Spike (e.g., Panic buying)
        Score < 1.0 = Supply Chain Slowness (e.g., Port closure)
        """
        prompt = f"""
        Analyze this supply chain news: "{news_text}"
        Return ONLY a JSON object with:
        - "score": (Float between 0.5 and 1.5, where 1.0 is neutral)
        - "category": (e.g., "Logistics", "Weather", "Economic")
        - "impact_summary": (Short 1-sentence explanation)
        """
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.url, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()['message']['content']
            return {"score": 1.0, "category": "None", "impact_summary": "Neutral state."}
        except:
            return {"score": 1.0, "category": "None", "impact_summary": "System offline."}

def get_mock_signals():
    """Simulates real-time external events for the dashboard."""
    return [
        {"event": "Severe Storm Warning: California Central Valley", "type": "Weather", "risk": "Medium"},
        {"event": "Port of Long Beach Labor Negotiations Stall", "type": "Logistics", "risk": "High"},
        {"event": "Regional SNAP Benefit Expansion Announced", "type": "Economic", "risk": "Positive"}
    ]
