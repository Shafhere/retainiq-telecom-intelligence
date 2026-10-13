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

    # Independent features (not tied to contract — otherwise they're proxies)
    support_calls = rng.poisson(lam=1.5, size=n)
    online_security = rng.choice([True, False], n, p=[0.35, 0.65])
    tech_support = rng.choice([True, False], n, p=[0.30, 0.70])
    days_since_last_login = rng.integers(0, 90, n)
    avg_monthly_gb = np.clip(
        np.where(internet == "fiber", rng.normal(180, 40, n),
        np.where(internet == "dsl", rng.normal(80, 30, n), rng.normal(5, 3, n))),
        0, 400,
    ).round(1)
    late_payments = rng.poisson(lam=0.8, size=n)

    # ---------------------------------------------------------------
    # MULTIPLICATIVE churn probability — each factor scales the base.
    # Much cleaner signal than sigmoid(logit) with a final multiplier.
    # ---------------------------------------------------------------
    base_prob = np.where(
        contract == "month-to-month", 0.28,
        np.where(contract == "one_year", 0.07, 0.02),
    )

    prob = base_prob.copy()
    prob *= np.where(tenure < 12, 2.0, 1.0)               # new customers churn more
    prob *= np.where(tenure > 48, 0.5, 1.0)               # loyal customers churn less
    prob *= np.where(internet == "fiber", 1.3, 1.0)       # premium → more likely to leave
    prob *= np.where(payment == "cash", 1.5, 1.0)         # cash payers less committed
    prob *= np.where(online_security, 0.6, 1.0)
    prob *= np.where(tech_support, 0.65, 1.0)
    prob *= np.where(support_calls >= 3, 1.5, 1.0)        # many complaints → risk
    prob *= np.where(days_since_last_login > 60, 1.6, 1.0)  # disengaged
    prob *= np.where(late_payments >= 2, 1.7, 1.0)        # payment issues

    prob = np.clip(prob, 0.005, 0.95)
    churn = (rng.random(n) < prob).astype(int)

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
        "days_since_last_login": days_since_last_login,
        "avg_monthly_gb": avg_monthly_gb,
        "late_payments": late_payments,
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