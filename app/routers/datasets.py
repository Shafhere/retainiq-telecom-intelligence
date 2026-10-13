from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import json
from datetime import datetime

from app.database import get_db
from app.models import DataVersion, ModelVersion
from app.config import settings
from app.services.ingest import (
    ingest_file, SUPPORTED_FORMATS, FEATURE_ALIASES,
    MODEL_NUMERIC, MODEL_CATEGORICAL, MODEL_BOOLEAN,
)
from app.services.trainer import train_models
from app.services.versioning import hash_dataframe
from app.services.predictor import predictor

router = APIRouter(prefix="/datasets", tags=["datasets"])

CSV_PATH = os.path.join(settings.data_dir, "telecom_customers.csv")


@router.get("/schema")
def schema_info():
    """What columns can be uploaded, and what aliases are accepted."""
    return {
        "supported_formats": sorted(SUPPORTED_FORMATS),
        "required_numeric": MODEL_NUMERIC,
        "required_categorical": MODEL_CATEGORICAL,
        "required_boolean": MODEL_BOOLEAN,
        "target_column": "churn (accepts Yes/No, True/False, 1/0)",
        "accepted_aliases": FEATURE_ALIASES,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    auto_retrain: bool = True,
    db: Session = Depends(get_db),
):
    """Upload CSV/XLSX/JSON/Parquet. Auto-maps columns, retrains, registers version."""
    try:
        content = await file.read()
        df, report = ingest_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {e}")

    # Compute content hash
    data_hash = hash_dataframe(df)

    # Check if this exact data was already uploaded
    existing = db.query(DataVersion).filter(DataVersion.content_hash == data_hash).first()
    if existing:
        return {
            "status": "duplicate",
            "message": "Identical dataset already uploaded. Skipping retrain.",
            "data_version_id": existing.id,
            "data_hash": data_hash[:16] + "...",
            "row_count": existing.row_count,
        }

    # Persist dataset to disk
    os.makedirs(settings.data_dir, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    # Register data version
    dv = DataVersion(
        content_hash=data_hash,
        filename=file.filename,
        row_count=len(df),
        churn_rate=float(df["churn"].mean()),
        is_active=True,
    )
    db.add(dv)
    db.commit()
    db.refresh(dv)

    result = {
        "status": "uploaded",
        "data_version_id": dv.id,
        "data_hash": data_hash[:16] + "...",
        "filename": file.filename,
        "rows": len(df),
        "churn_rate": round(float(df["churn"].mean()), 4),
        "column_mapping": report["mapped_columns"],
    }

    # Auto retrain
    if auto_retrain:
        try:
            meta = train_models(df)
            mv = ModelVersion(
                version=meta["version"],
                data_hash=data_hash,
                algorithm=meta["algorithm"],
                roc_auc=meta["metrics"]["roc_auc"],
                pr_auc=meta["metrics"]["pr_auc"],
                f1_score=meta["metrics"]["f1"],
                precision=meta["metrics"]["precision"],
                recall=meta["metrics"]["recall"],
                artifact_path=meta["model_path"],
                is_active=True,
            )

            # Deactivate older model versions
            db.query(ModelVersion).filter(ModelVersion.is_active == True).update(
                {"is_active": False}
            )
            db.add(mv)
            db.commit()

            predictor.reload()  # hot-swap the active model

            result["retrained"] = {
                "model_version": meta["version"],
                "algorithm": meta["algorithm"],
                "metrics": meta["metrics"],
            }
        except Exception as e:
            result["retrain_error"] = str(e)

    return result


@router.post("/retrain")
def retrain(db: Session = Depends(get_db)):
    """Retrain on the currently saved dataset."""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="No dataset on disk. Upload first.")
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    data_hash = hash_dataframe(df)
    meta = train_models(df)

    db.query(ModelVersion).filter(ModelVersion.is_active == True).update(
        {"is_active": False}
    )
    mv = ModelVersion(
        version=meta["version"],
        data_hash=data_hash,
        algorithm=meta["algorithm"],
        roc_auc=meta["metrics"]["roc_auc"],
        pr_auc=meta["metrics"]["pr_auc"],
        f1_score=meta["metrics"]["f1"],
        precision=meta["metrics"]["precision"],
        recall=meta["metrics"]["recall"],
        artifact_path=meta["model_path"],
        is_active=True,
    )
    db.add(mv)
    db.commit()
    predictor.reload()

    return {
        "status": "retrained",
        "model_version": meta["version"],
        "algorithm": meta["algorithm"],
        "metrics": meta["metrics"],
    }


@router.get("/versions")
def list_versions(db: Session = Depends(get_db)):
    """Show all uploaded data versions and trained model versions."""
    data_rows = db.query(DataVersion).order_by(DataVersion.id.desc()).all()
    model_rows = db.query(ModelVersion).order_by(ModelVersion.id.desc()).all()
    return {
        "data_versions": [
            {
                "id": d.id,
                "hash": d.content_hash[:16] + "...",
                "filename": d.filename,
                "row_count": d.row_count,
                "churn_rate": d.churn_rate,
                "is_active": d.is_active,
                "created_at": d.created_at.isoformat(),
            }
            for d in data_rows
        ],
        "model_versions": [
            {
                "id": m.id,
                "version": m.version,
                "algorithm": m.algorithm,
                "roc_auc": m.roc_auc,
                "f1_score": m.f1_score,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat(),
            }
            for m in model_rows
        ],
    }


@router.get("/active")
def active_model(db: Session = Depends(get_db)):
    """Which model is currently serving predictions."""
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if not mv:
        return {"error": "No active model registered"}
    return {
        "version": mv.version,
        "algorithm": mv.algorithm,
        "roc_auc": mv.roc_auc,
        "f1_score": mv.f1_score,
        "trained_at": mv.created_at.isoformat(),
    }