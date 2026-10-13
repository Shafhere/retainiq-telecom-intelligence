import io
import pandas as pd

SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

# Common aliases companies use for each required column
FEATURE_ALIASES = {
    "customer_id": ["customer_id", "customerid", "id", "customer", "cust_id"],
    "tenure_months": ["tenure_months", "tenure", "customer_tenure", "months_active", "tenure_in_months"],
    "monthly_charges": ["monthly_charges", "monthlycharges", "monthly_bill", "monthly_amount", "monthly_fee", "mrr"],
    "total_charges": ["total_charges", "totalcharges", "total_spent", "total_paid", "lifetime_value", "ltv"],
    "contract_type": ["contract_type", "contract", "plan_type", "plan"],
    "internet_service": ["internet_service", "internetservice", "internet", "internet_type"],
    "num_support_calls": ["num_support_calls", "support_calls", "num_complaints", "complaints", "support_tickets"],
    "payment_method": ["payment_method", "paymentmethod", "payment", "payment_type", "pay_method"],
    "has_online_security": ["has_online_security", "onlinesecurity", "online_security", "security_addon"],
    "has_tech_support": ["has_tech_support", "techsupport", "tech_support", "premium_support"],
    "days_since_last_login": ["days_since_last_login", "last_login_days", "days_inactive", "inactivity_days", "days_since_login"],
    "avg_monthly_gb": ["avg_monthly_gb", "avg_gb", "data_usage", "monthly_gb", "gb_used"],
    "late_payments": ["late_payments", "num_late_payments", "payment_delays", "missed_payments", "payment_late_count"],
    "churn": ["churn", "churned", "target", "label", "is_churn", "churn_flag", "exited"],
}

# Columns the trainer absolutely needs
REQUIRED_TARGET = "churn"

# Columns the model expects (from trainer.NUMERIC + CATEGORICAL + BOOLEAN)
MODEL_NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_calls", "days_since_last_login",
    "avg_monthly_gb", "late_payments",
]
MODEL_CATEGORICAL = ["contract_type", "internet_service", "payment_method"]
MODEL_BOOLEAN = ["has_online_security", "has_tech_support"]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_any_format(filename: str, content: bytes) -> pd.DataFrame:
    """Load a dataframe from CSV, Excel, JSON, or Parquet bytes."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    buf = io.BytesIO(content)

    if ext == ".csv":
        df = pd.read_csv(buf)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(buf)
    elif ext == ".json":
        df = pd.read_json(buf)
    elif ext == ".parquet":
        df = pd.read_parquet(buf)

    if df.empty:
        raise ValueError("Uploaded file has zero rows.")
    return df


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename incoming columns to canonical names using aliases."""
    df = df.copy()
    df.columns = [_normalize(c) for c in df.columns]

    rename_map = {}
    unmatched = []

    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in df.columns:
            continue  # already named correctly
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm in df.columns:
                rename_map[alias_norm] = canonical
                break

    df = df.rename(columns=rename_map)
    return df, rename_map


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """Check if the dataframe has everything the trainer needs."""
    required = MODEL_NUMERIC + MODEL_CATEGORICAL + MODEL_BOOLEAN + [REQUIRED_TARGET]
    missing = [c for c in required if c not in df.columns]
    optional_missing = []
    return (len(missing) == 0), missing, optional_missing


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric and boolean columns; fill safe defaults for optional ones."""
    df = df.copy()

    # Numeric
    for col in MODEL_NUMERIC:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Boolean — handle yes/no, true/false, 1/0, True/False
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"yes", "true", "1", "y", "t"}

    for col in MODEL_BOOLEAN:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].map(to_bool)

    # Categorical — force to string
    for col in MODEL_CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(str).str.strip()

    # Target — convert Yes/No or True/False to 1/0
    target = df[REQUIRED_TARGET]
    if target.dtype == object:
        target = target.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "churn": 1}
        )
    df[REQUIRED_TARGET] = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)

    return df


def ingest_file(filename: str, content: bytes) -> tuple[pd.DataFrame, dict]:
    """End-to-end: load → map columns → validate → coerce. Returns (df, report)."""
    df = load_any_format(filename, content)
    original_cols = list(df.columns)
    df, rename_map = map_columns(df)
    ok, missing, _ = validate_schema(df)

    report = {
        "original_columns": original_cols,
        "mapped_columns": rename_map,
        "missing_required": missing,
        "row_count": len(df),
        "is_valid": ok,
    }

    if not ok:
        raise ValueError(
            f"Missing required columns after mapping: {missing}. "
            f"Please rename them or provide an alias. "
            f"See /datasets/schema for accepted names."
        )

    df = coerce_types(df)
    report["final_columns"] = list(df.columns)
    report["churn_rate"] = float(df[REQUIRED_TARGET].mean())
    return df, report