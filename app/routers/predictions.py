from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.database import get_db
from app.models import PredictionAudit
from app.schemas import CustomerFeatures, PredictionResponse
from app.services.predictor import predictor

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict_customer(features: CustomerFeatures, db: Session = Depends(get_db)):
    try:
        customer = features.model_dump()
        result = predictor.predict(customer)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Persist audit trail
    audit = PredictionAudit(
        customer_id=result["customer_id"],
        model_version=result["model_version"],
        data_hash="live",
        churn_probability=result["churn_probability"],
        top_factors=json.dumps(result["top_factors"]),
        retention_offer=result["retention_offer"],
    )
    db.add(audit)
    db.commit()

    return result


@router.get("/recent")
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionAudit)
        .order_by(PredictionAudit.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "customer_id": r.customer_id,
            "churn_probability": r.churn_probability,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]