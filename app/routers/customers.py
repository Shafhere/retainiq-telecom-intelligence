from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import pandas as pd

from app.database import get_db
from app.config import settings
from app.services.predictor import predictor

router = APIRouter(prefix="/customers", tags=["customers"])

CSV_PATH = os.path.join(settings.data_dir, "telecom_customers.csv")


@router.get("/")
def list_customers(limit: int = 50, offset: int = 0):
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="No dataset found. Run the generator.")
    df = pd.read_csv(CSV_PATH).iloc[offset:offset + limit]
    return df.to_dict(orient="records")


@router.get("/count")
def customer_count():
    if not os.path.exists(CSV_PATH):
        return {"count": 0}
    df = pd.read_csv(CSV_PATH)
    return {"count": len(df), "churn_rate": float(df["churn"].mean())}


@router.post("/batch-score")
def batch_score(sample_size: int = 25):
    """Score a random sample of customers — demo endpoint."""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="No dataset found.")
    df = pd.read_csv(CSV_PATH).sample(n=sample_size, random_state=42)
    results = []
    for _, row in df.iterrows():
        try:
            res = predictor.predict(row.to_dict())
            results.append(res)
        except Exception as e:
            results.append({"customer_id": row.get("customer_id"), "error": str(e)})
    return results