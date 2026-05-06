import pandas as pd
import os
from experiment.LLM_x_mult_k import run as run_x_mult_k

CSV_PATH = "Results/LLM_response_x_mult_k.csv"
PRIMARY_KEYS = ["model", "k", "x"]

_cache = {}


def _validate_no_duplicate_primary_keys(df: pd.DataFrame, context: str) -> None:
    dup_mask = df.duplicated(subset=PRIMARY_KEYS, keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, PRIMARY_KEYS].drop_duplicates().head(5)
        raise ValueError(
            f"Duplicate primary keys found in {context} for keys {PRIMARY_KEYS}. "
            f"Sample duplicates: {dup_rows.to_dict(orient='records')}"
        )


def already_stored(model_id: str, k: int, x: int) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    df = pd.read_csv(CSV_PATH)
    mask = (df["model"] == model_id) & (df["k"] == k) & (df["x"] == x)
    match_count = int(mask.sum())
    if match_count > 1:
        raise ValueError(f"Expected at most one stored row for {(model_id, k, x)}, found {match_count}")
    return match_count == 1


def calc_and_store(model_id: str, pairs_k_x: list[tuple[int, int]]):
    """Runs experiment for all (k, x) pairs at once and appends results."""
    # Keep only unique requested pairs to avoid duplicate writes.
    unique_pairs = list(dict.fromkeys(pairs_k_x))
    result_df = run_x_mult_k(model_id=model_id, pairs_k_x=unique_pairs)
    _validate_no_duplicate_primary_keys(result_df, context="newly computed results")

    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        merged_df = pd.concat([existing_df, result_df], ignore_index=True)
        _validate_no_duplicate_primary_keys(merged_df, context=CSV_PATH)
        merged_df.to_csv(CSV_PATH, mode="w", header=True, index=False)
    else:
        result_df.to_csv(CSV_PATH, mode="w", header=True, index=False)


def get_result(model_id: str, k: int, x: int) -> float:
    df = pd.read_csv(CSV_PATH)
    mask = (df["model"] == model_id) & (df["k"] == k) & (df["x"] == x)
    matches = df.loc[mask, "p_err"]
    match_count = len(matches)
    if match_count != 1:
        raise ValueError(f"Expected exactly one match for {(model_id, k, x)}, found {match_count}")
    return float(matches.iloc[0])


def get_errors(model_id: str, pairs: list[tuple[int, int]]) -> dict:
    """
    pairs: list of (k, x) tuples
    Returns a dict mapping (model, k, x) -> float row
    """
    missing = [
        (k, x) for k, x in pairs
        if (model_id, k, x) not in _cache and not already_stored(model_id=model_id, k=k, x=x)
    ]

    if missing:
        calc_and_store(model_id=model_id, pairs_k_x=missing)

    for k, x in pairs:
        key = (model_id, k, x)
        if key not in _cache:
            _cache[key] = get_result(model_id, k, x)

    return {(model_id, k, x): _cache[(model_id, k, x)] for k, x in pairs}
