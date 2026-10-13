from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
import os
import json
import pandas as pd

from app.database import get_db
from app.models import PredictionAudit, ModelVersion
from app.config import settings

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    total_preds = db.query(PredictionAudit).count()

    high = db.query(PredictionAudit).filter(PredictionAudit.churn_probability >= 0.70).count()
    medium = db.query(PredictionAudit).filter(
        PredictionAudit.churn_probability >= 0.40,
        PredictionAudit.churn_probability < 0.70,
    ).count()
    low = total_preds - high - medium

    meta_path = os.path.join(settings.model_dir, "latest.json")
    metrics = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            metrics = meta.get("metrics", {})

    return {
        "total_predictions": total_preds,
        "risk_distribution": {"high": high, "medium": medium, "low": low},
        "model_metrics": metrics,
    }


@router.get("/risk-distribution")
def risk_distribution():
    csv_path = os.path.join(settings.data_dir, "telecom_customers.csv")
    if not os.path.exists(csv_path):
        return {"error": "No dataset"}
    df = pd.read_csv(csv_path)
    return {
        "contract_type": df.groupby("contract_type")["churn"].mean().round(4).to_dict(),
        "internet_service": df.groupby("internet_service")["churn"].mean().round(4).to_dict(),
        "payment_method": df.groupby("payment_method")["churn"].mean().round(4).to_dict(),
    }