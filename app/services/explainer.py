import pandas as pd
import numpy as np
import xgboost as xgb

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
    """
    Return top-k SHAP factors using XGBoost's built-in TreeSHAP.
    This avoids the `shap` library and its torch dependency.
    """
    # Get the preprocessor and classifier from the pipeline
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    # Transform the input row
    X_transformed = prep.transform(X_row)
    feature_names = prep.get_feature_names_out()

    # Use XGBoost's native SHAP implementation (pred_contribs=True)
    dmat = xgb.DMatrix(X_transformed, feature_names=list(feature_names))
    # The output is a matrix where the last column is the bias term
    shap_values = clf.get_booster().predict(dmat, pred_contribs=True)

    # The row of interest is the first row; exclude the bias term
    row_shap = shap_values[0][:-1]

    # Pair feature names with their SHAP values and original values
    pairs = list(zip(feature_names, row_shap, X_transformed[0]))
    # Sort by the absolute SHAP value, highest first
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