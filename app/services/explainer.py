import numpy as np
import shap
import pandas as pd


FEATURE_LABELS = {
    "tenure_months": "Customer tenure",
    "monthly_charges": "Monthly bill amount",
    "total_charges": "Total spend to date",
    "num_support_calls": "Recent support calls",
    "contract_type_month-to-month": "Month-to-month contract",
    "contract_type_one_year": "One-year contract",
    "contract_type_two_year": "Two-year contract",
    "internet_service_fiber": "Fiber internet",
    "internet_service_dsl": "DSL internet",
    "internet_service_none": "No internet service",
    "payment_method_credit_card": "Pays by credit card",
    "payment_method_bank_transfer": "Pays by bank transfer",
    "payment_method_mada": "Pays by Mada",
    "payment_method_cash": "Pays by cash",
    "has_online_security": "Has online security",
    "has_tech_support": "Has tech support",
}


def _label(raw_name: str) -> str:
    return FEATURE_LABELS.get(raw_name, raw_name.replace("_", " ").title())


def explain_prediction(pipeline, X_row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Return top-k SHAP factors with human-readable names and direction."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_transformed = prep.transform(X_row)
    feature_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_shap = shap_values[0]
    pairs = list(zip(feature_names, row_shap, X_transformed[0]))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    out = []
    for feat, sv, val in pairs[:top_k]:
        direction = "increases churn risk" if sv > 0 else "reduces churn risk"
        out.append({
            "feature": _label(feat),
            "raw_feature": feat,
            "shap_value": float(sv),
            "feature_value": str(val),
            "direction": direction,
        })
    return out