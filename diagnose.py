import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/telecom_customers.csv")

print("=" * 60)
print("DIAGNOSTIC")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Churn rate overall: {df['churn'].mean():.2%}")
print()

print("Churn rate by contract_type:")
print(df.groupby("contract_type")["churn"].agg(["mean", "count"]))
print()

print("Churn rate by internet_service:")
print(df.groupby("internet_service")["churn"].agg(["mean", "count"]))
print()

print("Churn rate by payment_method:")
print(df.groupby("payment_method")["churn"].agg(["mean", "count"]))
print()

print("Correlations with churn:")
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64, np.int32, np.float32, bool]:
        try:
            corr = df[col].astype(float).corr(df["churn"].astype(float))
            print(f"  {col:25s}: {corr:+.4f}")
        except Exception:
            pass

print()
print("Train a quick logistic regression to check signal:")

feature_cols = [c for c in df.columns if c not in ["customer_id", "city", "churn"]]
X = df[feature_cols]
y = df["churn"]

numeric = [c for c in feature_cols if df[c].dtype in [np.int64, np.float64, np.int32, np.float32]]
cat = [c for c in feature_cols if c not in numeric and df[c].dtype != bool]
bool_cols = [c for c in feature_cols if df[c].dtype == bool]

print(f"  numeric: {numeric}")
print(f"  categorical: {cat}")
print(f"  bool: {bool_cols}")

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("bool", "passthrough", bool_cols),
])
pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print(f"  LogisticRegression ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
print("=" * 60)