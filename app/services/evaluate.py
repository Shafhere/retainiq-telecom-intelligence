import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve


def threshold_sweep(y_true, y_proba) -> list[dict]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    out = []
    for i in range(0, len(thresholds), max(1, len(thresholds) // 20)):
        p = float(precisions[i])
        r = float(recalls[i])
        out.append({
            "threshold": float(thresholds[i]),
            "precision": p,
            "recall": r,
            "f1": float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
        })
    return out


def roc_points(y_true, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def calibration_data(y_true, y_proba, bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=bins, strategy="quantile")
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }