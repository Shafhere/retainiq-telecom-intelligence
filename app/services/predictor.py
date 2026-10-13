import os
import json
import joblib
import pandas as pd
from app.config import settings
from app.services.explainer import explain_prediction
from app.services.retention_advisor import retention_advisor


class Predictor:
    def __init__(self):
        self.pipeline = None
        self.meta = None
        self._load()

    def _load(self):
        meta_path = os.path.join(settings.model_dir, "latest.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No trained model found at {meta_path}. Run `python ml/train.py` first."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.pipeline = joblib.load(self.meta["model_path"])

    def reload(self):
        """Reload from disk — call after retraining to pick up new model."""
        self._load()

    @property
    def version(self) -> str:
        return self.meta.get("version", "unknown")

    def _risk_tier(self, prob: float) -> str:
        if prob >= 0.70:
            return "high"
        if prob >= 0.40:
            return "medium"
        return "low"

    def predict(self, customer: dict) -> dict:
        # Build the feature row exactly as training expects
        feature_cols = self.meta["feature_cols"]
        row = pd.DataFrame([{k: customer.get(k, 0) for k in feature_cols}])

        prob = float(self.pipeline.predict_proba(row)[0, 1])
        tier = self._risk_tier(prob)

        # SHAP explanation
        factors = explain_prediction(self.pipeline, row, top_k=5)

        # Bilingual retention offer
        offer_en, offer_ar = retention_advisor.generate(customer, prob, factors)

        return {
            "customer_id": customer.get("customer_id", "unknown"),
            "churn_probability": round(prob, 4),
            "risk_tier": tier,
            "top_factors": factors,
            "retention_offer": offer_en,
            "retention_offer_ar": offer_ar,
            "model_version": self.version,
        }


predictor = Predictor()