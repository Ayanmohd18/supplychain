# 📦 AI-Powered Supply Chain Forecasting Engine

An enterprise-grade, end-to-end demand forecasting system designed to optimize supply chain operations, reduce stockouts, and minimize overstock risks. Built around the renowned M5 Walmart Dataset, this system leverages state-of-the-art Deep Learning (Temporal Fusion Transformers) and ML pipelines to deliver high-fidelity predictions across tens of thousands of SKUs.

## 🚀 Key Features

- **State-of-the-Art Modeling**: Implements PyTorch-based Temporal Fusion Transformers (TFT) capable of capturing complex multi-horizon temporal dynamics, outperforming traditional ARIMA and XGBoost baselines by significant margins.
- **Explainable AI (XAI)**: Integrated SHAP (SHapley Additive exPlanations) values to interpret model decisions, quantifying the impact of features like pricing, seasonality, and special events on demand.
- **Production-Ready Dashboard**: A robust Streamlit web application providing a high-contrast, interactive control room for business stakeholders to explore forecasts, benchmark metrics, and monitor endpoint status.
- **Business Impact Simulation**: Translates mathematical metrics (WMAPE, Pinball Loss) into tangible business risks, actively calculating "Stockout Risk Days" and "Overstock Days" using P10-P90 prediction intervals.
- **MLOps & Automation**: Automated drift detection (Population Stability Index) and AWS-ready deployment scripts (EC2, S3, Lambda) for continuous recalibration and model lifecycle management.

## 📁 Project Architecture

The repository is structured to adhere to industry best practices for Machine Learning engineering:

```text
supplychain/
├── app.py                      # Interactive Streamlit Web Dashboard
├── data/                       # Dataset Storage Layer
│   ├── raw/                    # Raw M5 dataset files (calendar, sales, prices)
│   ├── interim/                # Intermediate transformed data
│   └── processed/              # Feature-engineered Parquet files for training
├── deploy/                     # Cloud Deployment Infrastructure
│   └── aws/
│       ├── ec2_setup.sh        # Automated EC2 instance configuration
│       ├── s3_upload.py        # Model artifact versioning to S3
│       └── lambda_retrain_trigger.py # CloudWatch metric trigger for automated retraining
├── notebooks/                  # R&D and Experimentation
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis & statistical validation
│   ├── 02_baseline_arima.ipynb # Traditional baseline establishment (ARIMA/Naive)
│   ├── 03_lstm_xgboost.ipynb   # Tree-based & sequential deep learning models
│   ├── 04_tft_training.ipynb   # Temporal Fusion Transformer implementation
│   ├── 05_shap_explainability.ipynb # Feature attribution and interpretation
│   └── 06_evaluation_dashboard.ipynb # Business metrics and comparative analysis
├── src/                        # Core Python Application Code
│   ├── api/                    # FastAPI endpoints for real-time inference
│   ├── data/                   # Data loaders and feature engineering pipelines
│   └── models/                 # Model architectures and baseline metrics
└── tests/                      # Comprehensive Unit & Integration Tests
    ├── test_api.py             # Endpoint validation
    ├── test_data.py            # Data integrity and loader testing
    └── test_models.py          # Algorithm shape and baseline validation
```

## 📊 Dataset: M5 Walmart Forecasting

The engine is trained on the robust M5 dataset, encompassing hierarchical sales data from Walmart:
- **Scope**: 42,840 distinct hierarchical time series.
- **Temporal Range**: 2011-01-29 to 2016-06-19.
- **Features Engineered**: 
  - *Temporal*: Day of week, month, year, cyclically encoded variables.
  - *Pricing*: Sell price, rolling price volatility.
  - *Events*: Holidays, SNAP days (Supplemental Nutrition Assistance Program) across states (CA, TX, WI).

## 🧠 Modeling Progression & Benchmarks

Our research and development pipeline systematically evaluated increasingly complex architectures to establish the optimal forecasting engine:

1. **ARIMA / Naive Baselines**: Established the foundational WMAPE benchmark.
2. **XGBoost**: Captured non-linear feature interactions but struggled with multi-horizon sequence mapping.
3. **LSTM**: Improved sequential pattern recognition.
4. **Temporal Fusion Transformer (TFT)**: The chosen architecture. TFT naturally handles static metadata, known future inputs (holidays), and unknown future inputs (sales), outputting highly accurate quantile predictions.
   - *Result*: Achieved a **15.3% reduction in WMAPE** vs the baseline.

## 🛠️ Installation & Usage

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ayanmohd18/supplychain.git
cd supplychain

# Initialize a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launching the Dashboard
Start the interactive Streamlit telemetry dashboard:
```bash
streamlit run app.py
```

### Running the API Server
Deploy the FastAPI backend for JSON-based inference:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## 🛡️ Testing & Validation

The repository includes a rigorous `pytest` suite ensuring data integrity, correct model dataset construction, and reliable API responses.
```bash
pytest tests/ -v
```

## ☁️ Cloud Deployment (AWS)

Production workflows are designed for AWS environments:
- **S3**: Model checkpoints and feature registries are versioned using `deploy/aws/s3_upload.py`.
- **Lambda & CloudWatch**: Automated drift detection monitors the WMAPE score. If degradation crosses the 0.60 threshold, `lambda_retrain_trigger.py` queues an EC2 retraining job.
- **EC2**: Containerized API hosting configured via `ec2_setup.sh`.

---
*Architected for predictive precision and supply chain resilience.*
