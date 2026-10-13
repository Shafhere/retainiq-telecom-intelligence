import os
import numpy as np
import pandas as pd
from app.config import settings


CONTRACTS = ["month-to-month", "one_year", "two_year"]
INTERNET = ["fiber", "dsl", "none"]
PAYMENTS = ["credit_card", "bank_transfer", "mada", "cash"]
CITIES = ["Riyadh", "Jeddah", "Dammam", "Mecca", "Medina", "Khobar"]


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(PAYMENTS, n, p=[0.35, 0.25, 0.30, 0.10])
    city = rng.choice(CITIES, n)

    base_charge = np.where(internet == "fiber", 320,
                    np.where(internet == "dsl", 180, 90))
    monthly = np.clip(base_charge + rng.normal(0, 25, n), 60, 500)
    total = np.clip(monthly * tenure + rng.normal(0, 200, n), 50, None)

    support_calls = rng.poisson(
        lam=np.where(contract == "month-to-month", 2.5, 1.0), size=n
    )
    online_security = rng.choice([True, False], n, p=[0.35, 0.65])
    tech_support = rng.choice([True, False], n, p=[0.30, 0.70])

    logit = (
        -0.9 * (tenure < 12)
        + 1.4 * (contract == "month-to-month")
        - 0.8 * (contract == "two_year")
        + 0.9 * (internet == "fiber")
        + 0.05 * support_calls
        + 0.6 * (monthly > 250)
        + 0.5 * (payment == "cash")
        - 0.7 * online_security
        - 0.6 * tech_support
        + rng.normal(0, 0.6, n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob * 0.55).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "city": city,
        "tenure_months": tenure,
        "monthly_charges": monthly.round(2),
        "total_charges": total.round(2),
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "num_support_calls": support_calls,
        "has_online_security": online_security,
        "has_tech_support": tech_support,
        "churn": churn,
    })


def save_dataset(df: pd.DataFrame, path: str = None) -> str:
    if path is None:
        os.makedirs(settings.data_dir, exist_ok=True)
        path = os.path.join(settings.data_dir, "telecom_customers.csv")
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = generate_customers()
    path = save_dataset(df)
    print(f"Saved {len(df)} customers to {path}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Columns: {list(df.columns)}")