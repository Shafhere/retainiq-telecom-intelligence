# 📡 RetainIQ — Telecom Customer Intelligence Platform

End-to-end churn prediction platform with SHAP explainability and bilingual (English / العربية) retention offers. Built for telecom operators in Saudi Arabia and the Middle East.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)

[Live Demo] : https://youtu.be/3UnOtNmFVHQ

## The Problem

A telecom operator with 5 million subscribers loses ~2% per month to churn — roughly **SAR 300M per year** in lost lifetime value. Predicting *who* will churn is only half the answer. The other half is **what to offer them to stay**.

RetainIQ solves both:
- 🎯 **Predicts churn** with an XGBoost + LightGBM ensemble
- 🔍 **Explains why** with per-customer SHAP factors
- 💬 **Prescribes retention** with a bilingual LLM advisor grounded in the company's retention policy
- 📦 **Ingests any format** — CSV, Excel, JSON, Parquet — with automatic column mapping

## Architecture

```
Upload (CSV/Excel/JSON/Parquet)
    ↓
Column auto-mapping + validation
    ↓
SHA-256 content hashing (versioning)
    ↓
Train XGBoost + LightGBM ensemble
    ↓
Evaluate (ROC-AUC, PR-AUC, F1, calibration)
    ↓
Register model version in DB
    ↓
Serve predictions
    ↓
SHAP explainability + LLM retention advisor (EN/AR)
    ↓
FastAPI → Streamlit dashboard
```

## Features

| Feature | Detail |
|---|---|
| **Ensemble ML** | XGBoost + LightGBM with stratified 5-fold CV |
| **Class imbalance** | `scale_pos_weight` for XGBoost, `class_weight="balanced"` for LightGBM |
| **Explainability** | Per-customer SHAP values with human-readable labels |
| **Bilingual LLM** | Groq Llama 3.3 generates retention offers in English + Arabic |
| **RAG grounding** | Retention policy in prompt prevents hallucinated discounts |
| **Multi-format upload** | CSV, XLSX, XLS, JSON, Parquet |
| **Column auto-mapping** | Understands 100+ common telecom column aliases |
| **Versioning** | SHA-256 content hash for data; every model version registered in DB |
| **Audit trail** | Every prediction logged with model version + factors |
| **Hot-swap retrain** | Upload new data → retrains → swaps active model without restart |
| **SageMaker path** | Production training script for AWS |

## Tech Stack

Python 3.12 · FastAPI · Streamlit · XGBoost · LightGBM · SHAP · scikit-learn · Pandas · Plotly · LangChain · Groq · SQLAlchemy · SQLite/PostgreSQL · Docker · AWS SageMaker

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Shafhere/retainiq-telecom-intelligence.git
cd retainiq-telecom-intelligence
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com/keys
```

### 3. Generate synthetic data and train

```bash
python -m app.services.data_generator
python ml/train.py
```

### 4. Run

**Terminal 1 — API:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 — Dashboard:**
```bash
streamlit run dashboard/app.py
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### 5. Docker (alternative)

```bash
docker compose up --build
```

## Model Performance

Trained on 5,000 synthetic customers with realistic churn patterns.

| Metric | XGBoost | LightGBM |
|---|---|---|
| ROC-AUC | 0.88 | 0.87 |
| PR-AUC | 0.72 | 0.70 |
| F1 | 0.68 | 0.66 |
| Precision | 0.70 | 0.68 |
| Recall | 0.66 | 0.64 |

*(Replace with your actual metrics from `ml/models/latest.json`.)*

## Upload Your Own Data

Go to the **Upload & Version** tab in the dashboard:

1. Upload CSV / Excel / JSON / Parquet
2. System auto-maps columns using 100+ recognized aliases
3. Data is hashed and registered as a new version
4. Model retrains on the new data
5. New model version is registered and becomes active
6. Previous model is deactivated (still in history)

Accepted column aliases: see `GET /datasets/schema` in the API docs, or expand the "Show accepted column aliases" section in the dashboard.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predictions/predict` | Score a single customer |
| GET | `/predictions/recent` | Recent prediction audit trail |
| GET | `/customers/` | List customers (paginated) |
| GET | `/customers/count` | Total customers + churn rate |
| POST | `/customers/batch-score` | Score random sample |
| GET | `/analytics/summary` | Dashboard metrics |
| GET | `/analytics/risk-distribution` | Churn by segment |
| POST | `/datasets/upload` | Upload new dataset + retrain |
| POST | `/datasets/retrain` | Retrain on current data |
| GET | `/datasets/versions` | Data + model version history |
| GET | `/datasets/active` | Currently serving model |
| GET | `/datasets/schema` | Accepted columns + aliases |

Full interactive docs at http://localhost:8000/docs

## Project Structure

```
retainiq-telecom-intelligence/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py                # Versioning + audit tables
│   ├── schemas.py
│   ├── services/
│   │   ├── data_generator.py    # Synthetic data
│   │   ├── ingest.py            # Multi-format loader + column mapping
│   │   ├── versioning.py        # SHA-256 content hashing
│   │   ├── trainer.py           # XGBoost + LightGBM ensemble
│   │   ├── evaluate.py          # ROC/PR curves, calibration
│   │   ├── explainer.py         # SHAP + human-readable labels
│   │   ├── retention_advisor.py # Bilingual LLM offer generator
│   │   └── predictor.py         # End-to-end inference
│   └── routers/
│       ├── datasets.py          # Upload, retrain, versions
│       ├── predictions.py       # Single customer scoring
│       ├── customers.py         # List, count, batch-score
│       └── analytics.py         # Dashboard metrics
├── dashboard/
│   └── app.py                   # Streamlit UI
├── ml/
│   ├── train.py                 # Local training entrypoint
│   └── models/                  # Saved artifacts
├── sagemaker/
│   ├── train_sagemaker.py       # AWS SageMaker launcher
│   └── README.md
├── data/                        # CSV uploads
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## AWS SageMaker Deployment

See [`sagemaker/README.md`](sagemaker/README.md) for the production training workflow.

## Demo

[Add your YouTube demo link here]

## License

MIT
