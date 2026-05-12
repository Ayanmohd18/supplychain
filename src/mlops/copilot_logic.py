import pandas as pd
import numpy as np
import requests
import json

class CopilotEngine:
    """
    The reasoning engine behind the Supply Chain Copilot.
    Now integrated with OLLAMA for local LLM power.
    """
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url}/api/chat"

    def get_context_string(self):
        """Creates a text-based summary of the current supply chain state."""
        return """
        CURRENT SYSTEM STATE:
        - Model: Temporal Fusion Transformer (TFT)
        - Performance: WMAPE 0.24 (15.3% improvement vs baseline)
        - Active Risks: 3 items in CA_1 (Critical: FOODS_3_090_CA_1)
        - Inventory Optimization: Optimized at 95% service level.
        - Simulation Engine: Active (Elasticity -1.5)
        """

    def get_response(self, query):
        context = self.get_context_string()
        system_prompt = f"""
        You are the 'Nothing OS Supply Chain Copilot', an elite AI logistics analyst.
        Context: {context}
        
        Instructions:
        1. Be concise and professional (Nothing OS aesthetic).
        2. Use the provided context to answer data-specific questions.
        3. If asked about stockouts, mention the critical risk in FOODS_3_090_CA_1.
        4. If Ollama is working, give a detailed technical insight.
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return {
                    "role": "assistant",
                    "content": result['message']['content'],
                    "action": "OLLAMA ENGINE ACTIVE"
                }
            else:
                return self._fallback_response(query)
        except Exception:
            return self._fallback_response(query)

    def _fallback_response(self, query):
        """Rule-based fallback if Ollama is not running."""
        query = query.lower()
        if "stockout" in query or "risk" in query:
            return {
                "role": "assistant",
                "content": "◆ OLLAMA OFFLINE. FALLBACK ANALYTICS ACTIVE...\n\nHigh-priority stockout risk detected in `FOODS_3_090_CA_1`. Replenishment recommended within 48 hours.",
                "action": "Check Alert Tab"
            }
        else:
            return {
                "role": "assistant",
                "content": "I am currently running in 'Lite Mode' (Ollama not detected). Please ensure Ollama is running locally with 'ollama run llama3' for full reasoning capabilities.",
                "action": "Launch Ollama"
            }

def initialize_chat():
    return [
        {"role": "assistant", "content": "SYSTEM ONLINE. OLLAMA AGENT INITIALIZED. How can I assist with your logistics strategy today?"}
    ]
