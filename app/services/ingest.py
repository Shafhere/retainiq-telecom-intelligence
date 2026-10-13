import io
import re
from difflib import SequenceMatcher
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
    "num_support_calls": ["num_support_calls", "support_calls", "num_complaints", "complaints", "support_tickets", "support_call_count", "recent_support_calls"],
    "payment_method": ["payment_method", "paymentmethod", "payment", "payment_type", "pay_method"],
    "has_online_security": ["has_online_security", "onlinesecurity", "online_security", "security_addon"],
    "has_tech_support": ["has_tech_support", "techsupport", "tech_support", "premium_support"],
    "days_since_last_login": ["days_since_last_login", "last_login_days", "days_inactive", "inactivity_days", "days_since_login", "days_since_login", "last_login"],
    "avg_monthly_gb": ["avg_monthly_gb", "avg_gb", "data_usage", "monthly_gb", "gb_used", "average_gb", "avg_data_usage"],
    "late_payments": ["late_payments", "num_late_payments", "payment_delays", "missed_payments", "payment_late_count", "late_payment_count"],
    "churn": ["churn", "churned", "target", "label", "is_churn", "churn_flag", "exited"],
}

# Columns the model expects (from trainer.NUMERIC + CATEGORICAL + BOOLEAN)
MODEL_NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_calls", "days_since_last_login",
    "avg_monthly_gb", "late_payments",
]
MODEL_CATEGORICAL = ["contract_type", "internet_service", "payment_method"]
MODEL_BOOLEAN = ["has_online_security", "has_tech_support"]
REQUIRED_TARGET = "churn"


def _normalize(name: str) -> str:
    """Lowercase, split CamelCase, replace spaces/dashes with underscores."""
    # Split CamelCase: SupportCalls -> Support_Calls, avgGB -> avg_GB
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)   # XMLParser -> XML_Parser
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s)                            # collapse __
    return s


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
    """Rename incoming columns to canonical names using aliases + fuzzy fallback."""
    df = df.copy()
    df.columns = [_normalize(c) for c in df.columns]

    rename_map = {}
    used_canonical = set()

    # -------- Pass 1: exact alias match --------
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in df.columns:
            used_canonical.add(canonical)
            continue
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm in df.columns:
                rename_map[alias_norm] = canonical
                used_canonical.add(canonical)
                break

    # -------- Pass 2: fuzzy match remaining columns --------
    remaining_cols = [c for c in df.columns if c not in rename_map and c not in used_canonical]
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in used_canonical:
            continue
        best_score = 0.0
        best_col = None
        for col in remaining_cols:
            for alias in aliases:
                alias_norm = _normalize(alias)
                score = SequenceMatcher(None, col, alias_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_col = col
        if best_score >= 0.82 and best_col:
            rename_map[best_col] = canonical
            used_canonical.add(canonical)
            remaining_cols.remove(best_col)

    df = df.rename(columns=rename_map)
    return df, rename_map

def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """Check if the dataframe has everything the trainer needs."""
    required = MODEL_NUMERIC + MODEL_CATEGORICAL + MODEL_BOOLEAN + [REQUIRED_TARGET]
    missing = [c for c in required if c not in df.columns]
    optional_missing = []
    return (len(missing) == 0), missing, optional_missing


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric, boolean, and target columns; handle many value formats."""
    df = df.copy()

    # Numeric
    for col in MODEL_NUMERIC:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Boolean
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

    # Categorical
    for col in MODEL_CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(str).str.strip()

    # ----- Target (churn) — ROBUST conversion -----
    target = df[REQUIRED_TARGET]
    print(f"[ingest] target dtype={target.dtype}, sample={target.head(5).tolist()}")

    # If already numeric, just coerce
    if pd.api.types.is_numeric_dtype(target):
        df[REQUIRED_TARGET] = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
        print(f"[ingest] numeric target — rate={df[REQUIRED_TARGET].mean():.3f}")
        return df

    # Otherwise treat as string and map
    target_str = target.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0, "1.0": 1, "0.0": 0,
        "churn": 1, "churned": 1, "exited": 1, "left": 1,
        "active": 0, "retained": 0, "stayed": 0, "no_churn": 0,
        "y": 1, "n": 0, "t": 1, "f": 0,
        "yes ": 1, " no": 0,  # double safety
    }

    mapped = target_str.map(mapping)
    unmapped = mapped.isna().sum()
    if unmapped > 0:
        print(f"[ingest] WARNING: {unmapped} unmapped target values. Sample: {target_str[mapped.isna()].head(5).tolist()}")

    # If mapping failed entirely, try numeric conversion
    if mapped.isna().all():
        print("[ingest] Mapping failed — trying numeric conversion")
        mapped = pd.to_numeric(target_str, errors="coerce")

    df[REQUIRED_TARGET] = mapped.fillna(0).astype(int)
    print(f"[ingest] final churn rate={df[REQUIRED_TARGET].mean():.3f}, positives={int(df[REQUIRED_TARGET].sum())}")
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