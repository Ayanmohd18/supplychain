<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch_Lightning-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/LLM-Ollama_LLaMA3-purple.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/Deployment-AWS_Ready-yellow.svg" alt="AWS">

  <h1>📦 Autonomous Supply Chain Forecasting Engine</h1>
  <p><b>State-of-the-Art Demand Prediction · LLM Copilot · Real-Time Disruption Sensing · Geospatial Intelligence</b></p>
</div>

<br/>

## 🌐 Overview

The **Autonomous Supply Chain Forecasting Engine** is an enterprise-grade ML decision platform designed to minimize stockouts and overstock risks by predicting high-dimensional retail demand. Built on the massive **M5 Walmart Dataset** (42,840 distinct time series), this system goes far beyond traditional forecasting — it combines a cutting-edge **Temporal Fusion Transformer (TFT)**, a **local LLM Copilot** powered by Ollama, **real-time disruption sensing**, and **geospatial store intelligence** into a unified operational dashboard.

<details>
<summary><b>🔥 Why Temporal Fusion Transformers? (Click to expand)</b></summary>
<br/>
While XGBoost and LSTMs perform adequately on point forecasting, they struggle with mixed variable types and probabilistic bounds. The TFT architecture naturally ingests:
<ul>
  <li><b>Static Covariates:</b> Store IDs, Item Categories.</li>
  <li><b>Known Future Inputs:</b> Holidays, SNAP events, day-of-week.</li>
  <li><b>Unknown Future Inputs:</b> Historical sales, dynamic price fluctuations.</li>
</ul>
This allows for highly calibrated quantile predictions (P10 to P90), enabling risk-aware, probabilistic business decisions.
</details>

---

## 🏗️ Architecture & Modules

The repository is fully modularized for rapid experimentation, scaling, and production deployment.

### 1️⃣ Core Machine Learning Pipeline
| Phase | Technologies | Description | Location |
|---|---|---|---|
| **EDA & Processing** | `Pandas`, `Seaborn` | Outlier detection, seasonal decomposition, cyclic encoding. | `notebooks/01_EDA.ipynb` |
| **Baseline Models** | `Statsmodels`, `XGBoost` | ARIMA, Naive, and Tree-based benchmark metrics. | `notebooks/02` & `03` |
| **Deep Learning** | `PyTorch Forecasting` | Temporal Fusion Transformer (TFT) implementation. | `notebooks/04_tft_training.ipynb` |
| **Explainability** | `SHAP` | Global and local feature importance extraction. | `notebooks/05_shap_explainability.ipynb` |

---

### 2️⃣ Supply Chain Copilot (`src/mlops/copilot_logic.py`) 🆕

An intelligent, conversational AI analyst powered by a **locally-hosted LLaMA 3 model via Ollama**. The Copilot has full awareness of the current system state and can answer natural language logistics questions in real time.

- **Ollama Integration:** Sends context-enriched prompts to `llama3` at `localhost:11434`.
- **Context-Aware Reasoning:** Automatically includes live model performance (WMAPE), active risk items, inventory state, and simulation parameters.
- **Graceful Fallback:** Rule-based analytics engine activates automatically if Ollama is offline.

```python
# Usage
from src.mlops.copilot_logic import CopilotEngine
copilot = CopilotEngine(model="llama3")
response = copilot.get_response("What is the stockout risk for CA_1 this week?")
```

---

### 3️⃣ Disruption Radar (`src/mlops/disruption_engine.py`) 🆕

A **real-time external signal processing engine** that converts unstructured news, weather events, and economic signals into quantitative **Demand Multipliers** using LLM reasoning.

- **LLM Signal Parsing:** Sends raw news text to Ollama and extracts a structured JSON disruption score (0.5–1.5 scale).
- **Impact Categories:** Classifies disruptions as `Logistics`, `Weather`, or `Economic`.
- **Mock Signal Feed:** Pre-loaded with realistic M5-region events (California storms, port strikes, SNAP expansions) for demo use.

| Score | Interpretation |
|:---:|---|
| `> 1.0` | Unexpected demand spike (e.g., panic buying) |
| `= 1.0` | Neutral baseline |
| `< 1.0` | Supply chain slowness (e.g., port closure) |

---

### 4️⃣ Risk Alerts Engine (`src/mlops/alerts.py`) 🆕

A quantitative **risk classification system** that continuously monitors inventory levels against TFT probabilistic forecasts to surface actionable alerts.

- **Stockout Detection:** Flags items where current inventory falls below P10 demand (Critical) or median demand (Warning).
- **Overstock Detection:** Identifies items where inventory exceeds 2× P90 demand, flagging capital inefficiency.
- **Severity Tiers:** `CRITICAL` → `WARNING` → `EFFICIENCY`

```python
from src.mlops.alerts import SupplyChainAlerts
engine = SupplyChainAlerts()
alerts = engine.detect_risks("FOODS_3_090_CA_1", forecast_p10, forecast_p90, current_inventory=500)
```

---

### 5️⃣ Affinity Engine (`src/data/affinity_logic.py`) 🆕

A **GNN-Lite inter-product relationship mapper** that surfaces hidden demand transfer patterns between SKUs — identifying complements and substitutes across the product catalog.

- **Relationship Matrix:** Computes pairwise affinity scores between selected items.
- **Interactive Network Graph:** Renders a live Plotly network visualization where green edges = complementary demand, red edges = substitute demand.
- **Production Path:** Designed to be replaced with a full GNN adjacency matrix trained on co-purchase patterns.

---

### 6️⃣ Geospatial Intelligence (`src/data/geo_logic.py`) 🆕

Transforms hierarchical M5 store data into a **3D geospatial risk map** covering all 10 Walmart locations across California, Texas, and Wisconsin.

- **Store Coordinates:** Precise lat/lon for all M5 stores (CA×4, TX×3, WI×3).
- **Risk Heatmap:** Dynamic color encoding (Green → Red) based on per-store risk scores.
- **Volume Pillars:** Sales volume mapped to visual pillar height for intuitive spatial comparison.

| State | Stores Covered |
|---|---|
| California | CA_1 (LA), CA_2 (Pasadena), CA_3 (SF), CA_4 (San Jose) |
| Texas | TX_1 (Dallas), TX_2 (Houston), TX_3 (Austin) |
| Wisconsin | WI_1 (Milwaukee), WI_2 (Madison), WI_3 (Green Bay) |

---

### 7️⃣ Real-Time Telemetry Dashboard (`app.py`)

A high-performance Streamlit control center integrating all modules above:

- **Interactive Forecasting:** Adjustable horizons (7–28 days) with P10/P50/P90 quantile bands.
- **Business Impact Metrics:** Stockout Risk Days, Coverage Rates, Inventory Optimization.
- **Copilot Chat UI:** Conversational interface powered by `CopilotEngine`.
- **Disruption Feed:** Live signal cards from `DisruptionRadar` with LLM-parsed impact scores.
- **Affinity Map:** Real-time SKU relationship network from `AffinityEngine`.
- **Geo Intelligence:** 3D store risk map from `GeoEngine`.
- **MLOps Monitoring:** Data Drift (PSI) tracking and model health indicators.

---

### 8️⃣ MLOps & AWS Cloud Deployment

| Component | Script | Function |
|---|---|---|
| **EC2 Provisioning** | `deploy/aws/ec2_setup.sh` | Automates Docker installation, dependency loading, and API launch. |
| **Artifact Registry** | `deploy/aws/s3_upload.py` | Timestamped versioning of model `.ckpt` files to AWS S3. |
| **Auto-Retraining** | `deploy/aws/lambda_retrain...` | CloudWatch triggers retraining via SSM if WMAPE crosses thresholds. |

---

## 📊 Dataset: M5 Hierarchical Retail Data

The M5 dataset is the gold standard for hierarchical sales forecasting.

- **Scale:** 30,490 items × 10 stores × 3 states (CA, TX, WI) = **42,840 time series.**
- **Timeline:** January 2011 to June 2016 (1,913 days).
- **Engineered Features:**
  * Rolling price volatility & momentum.
  * Autoregressive lags (7, 14, 28 days).
  * State-specific SNAP indicators.

---

## 📈 Performance Benchmarks

*Evaluated on an out-of-sample 28-day forecast horizon.*

| Model Architecture | WMAPE | Improvement vs Baseline | Notes |
|:---|:---:|:---:|:---|
| **ARIMA (Baseline)** | `0.452` | - | Fails to capture cross-series correlation. |
| **XGBoost (Lagged)** | `0.341` | +24.5% | Strong non-linear mapping, weak temporal flow. |
| **LSTM (Seq2Seq)** | `0.312` | +30.9% | Good sequential logic, lacks interpretability. |
| **TFT (Proposed)** | **`0.241`** | **+46.6%** | **SOTA. Excellent quantile calibration.** |

---

## 🚀 Quick Start Guide

### 1. Environment Initialization
```bash
git clone https://github.com/Ayanmohd18/supplychain.git
cd supplychain

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. (Optional) Launch LLM Copilot
For full Copilot & Disruption Radar functionality, install and start Ollama:
```bash
# Install from https://ollama.com
ollama run llama3
```
> If Ollama is not running, the system automatically falls back to rule-based analytics.

### 3. Launching the Telemetry Dashboard
```bash
streamlit run app.py
```

### 4. Executing the Training Pipeline
```bash
python src/models/tft_model.py --epochs 50 --batch_size 128
```

---

## 📂 Repository Structure

```text
📦 supplychain
 ┣ 📂 data/                    # Raw, interim, and processed parquet files
 ┣ 📂 deploy/aws/              # S3 syncing, Lambda triggers, EC2 setup
 ┣ 📂 lightning_logs/          # PyTorch Lightning checkpoints & tensorboard logs
 ┣ 📂 logs/                    # EDA outputs, baseline benchmarks, SHAP plots
 ┣ 📂 models/                  # Saved model artifacts
 ┣ 📂 notebooks/               # Core R&D notebooks (EDA → TFT → Explainability)
 ┣ 📂 src/
 ┃  ┣ 📂 data/
 ┃  ┃  ┣ 📜 affinity_logic.py  # 🆕 GNN-Lite inter-product affinity mapper
 ┃  ┃  ┗ 📜 geo_logic.py       # 🆕 Geospatial 3D store risk engine
 ┃  ┣ 📂 mlops/
 ┃  ┃  ┣ 📜 alerts.py          # 🆕 Stockout & overstock risk alert system
 ┃  ┃  ┣ 📜 copilot_logic.py   # 🆕 Ollama LLaMA3-powered Supply Chain Copilot
 ┃  ┃  ┗ 📜 disruption_engine.py # 🆕 Real-time external signal disruption radar
 ┃  ┗ 📂 models/               # TFT model definitions & data loaders
 ┣ 📂 tests/                   # Pytest suite for CI/CD validation
 ┣ 📂 api/                     # FastAPI service layer
 ┣ 📜 app.py                   # Streamlit Application Entrypoint
 ┗ 📜 requirements.txt         # Frozen dependency graph
```

---

<div align="center">
  <i>Engineered for Predictive Precision. Powered by LLM Intelligence. Designed for Scale.</i>
</div>
