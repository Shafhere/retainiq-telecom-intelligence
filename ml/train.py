import os
import sys
import json
import pandas as pd

# make `app` importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_generator import generate_customers, save_dataset
from app.services.trainer import train_models
from app.services.versioning import hash_dataframe
from app.config import settings


def main():
    print("=" * 60)
    print("RetainIQ — Training Pipeline")
    print("=" * 60)

    # 1. Load or generate data
    csv_path = os.path.join(settings.data_dir, "telecom_customers.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("No dataset found — generating...")
        df = generate_customers()
        save_dataset(df)

    print(f"Rows: {len(df)}")
    print(f"Churn rate: {df['churn'].mean():.2%}")

    # 2. Version the data
    data_hash = hash_dataframe(df)
    print(f"Data hash: {data_hash[:16]}...")

    # 3. Train
    print("\nTraining models...")
    meta = train_models(df)

    print(f"\nBest model: {meta['algorithm'].upper()}")
    m = meta["metrics"]
    print(f"  ROC-AUC : {m['roc_auc']:.4f}   (CV {m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.4f})")
    print(f"  PR-AUC  : {m['pr_auc']:.4f}")
    print(f"  F1      : {m['f1']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall  : {m['recall']:.4f}")
    cm = m["confusion_matrix"]
    print(f"  Confusion: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

    # 4. Compare all candidates
    print("\nAll candidates:")
    for name, metrics in meta["all_results"].items():
        print(f"  {name:10s} ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

    # 5. Persist versioning info
    version_file = os.path.join(settings.model_dir, "version_history.json")
    history = []
    if os.path.exists(version_file):
        with open(version_file) as f:
            history = json.load(f)
    history.append({
        "version": meta["version"],
        "data_hash": data_hash,
        "algorithm": meta["algorithm"],
        "metrics": meta["metrics"],
        "trained_at": meta["trained_at"],
    })
    with open(version_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nModel saved: {meta['model_path']}")
    print(f"Metadata   : {os.path.join(settings.model_dir, 'latest.json')}")
    print(f"History    : {version_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()