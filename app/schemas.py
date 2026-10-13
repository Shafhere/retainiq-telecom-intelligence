from pydantic import BaseModel
from datetime import datetime


class CustomerFeatures(BaseModel):
    customer_id: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    num_support_calls: int
    payment_method: str = "credit_card"
    has_online_security: bool = False
    has_tech_support: bool = False


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_factors: list[dict]
    retention_offer: str
    retention_offer_ar: str
    model_version: str


class ModelMetrics(BaseModel):
    version: str
    roc_auc: float
    pr_auc: float
    f1_score: float
    precision: float
    recall: float
    created_at: datetime

    class Config:
        from_attributes = True