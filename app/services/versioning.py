import hashlib


def hash_dataframe(df) -> str:
    """Deterministic SHA-256 of a dataframe's content."""
    row_hashes = []
    for _, row in df.iterrows():
        row_str = "|".join(str(v) for v in row.values)
        row_hashes.append(hashlib.sha256(row_str.encode()).hexdigest()[:16])
    combined = "".join(sorted(row_hashes))
    return hashlib.sha256(combined.encode()).hexdigest()