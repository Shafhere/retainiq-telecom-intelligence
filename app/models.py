from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from datetime import datetime
from app.database import Base


class DataVersion(Base):
    __tablename__ = "data_versions"
    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String, unique=True, index=True)
    filename = Column(String)
    row_count = Column(Integer)
    churn_rate = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    data_hash = Column(String)
    algorithm = Column(String)
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    artifact_path = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    tenure_months = Column(Integer)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    contract_type = Column(String)
    internet_service = Column(String)
    num_support_calls = Column(Integer)
    churn_probability = Column(Float, default=0.0)
    risk_tier = Column(String, default="low")
    last_scored_at = Column(DateTime, default=datetime.utcnow)


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String)
    model_version = Column(String)
    data_hash = Column(String)
    churn_probability = Column(Float)
    top_factors = Column(Text)
    retention_offer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)