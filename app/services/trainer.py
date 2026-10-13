import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from app.config import settings

NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_calls", "days_since_last_login",
    "avg_monthly_gb", "late_payments",
]
CATEGORICAL = ["contract_type", "internet_service", "payment_method"]
BOOLEAN = ["has_online_security", "has_tech_support"]
DROP = ["customer_id", "city", "churn"]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ]
    )


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_test, y_pred_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_models(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
    feature_cols = [c for c in df.columns if c not in DROP]
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor()

    # Compute class imbalance ratio (negatives / positives)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = float(neg) / float(pos)
    print(f"Class balance — neg: {neg}, pos: {pos}, scale_pos_weight: {scale:.2f}")

    candidates = {
        "xgboost": XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
    }

    results = {}
    best_name, best_pipe, best_auc = None, None, -1.0

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
        results[name] = metrics
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_pipe = pipe

    # Save artifacts
    os.makedirs(settings.model_dir, exist_ok=True)
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    model_path = os.path.join(settings.model_dir, f"{best_name}_{version}.joblib")
    joblib.dump(best_pipe, model_path)

    meta = {
        "version": version,
        "algorithm": best_name,
        "model_path": model_path,
        "feature_cols": feature_cols,
        "test_size": test_size,
        "seed": seed,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": results[best_name],
        "all_results": results,
    }
    with open(os.path.join(settings.model_dir, "latest.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta