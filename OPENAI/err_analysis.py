import json
import pandas as pd
from pathlib import Path
from OPENAI.parsers import parse_err_extraction, parse_with_mapping


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_log(log_path: str = "usage_log.jsonl") -> pd.DataFrame:
    rows = []
    with Path(log_path).open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ── Explosion: one log entry → one row per (c, x) ────────────────────────────

def explode_entry(row: pd.Series, mapping_path: str = "err_mapping.json") -> list[dict]:
    metadata = row["metadata"]
    C = metadata["C"]
    x = metadata["x"]
    ground_truth = metadata["ground_truth"]   # list[bool]

    print(row['response'])
    sucess, predicted = parse_with_mapping(
        text=row["response"],
        parse_fn=parse_err_extraction,
        mapping_path=mapping_path,
    )
    if not sucess:
        return [
            {
                "c":                c,
                "x":                x,
                "predicted":        None,
                "ground_truth":     ground_truth[i],
                "correct":          False,
                "model":            row["model"],
                "reasoning_effort": row["reasoning_effort"],
                "timestamp":        row["timestamp"],
                "parse_successful":   False,
                "status": "parse_failure",
                "raw_response": row["response"]
            }
            for i, c in enumerate(C)]

    if len(predicted) != len(C):
        print(row['response'], predicted)
        raise ValueError(
            f"Length mismatch for x={x}: "
            f"predicted has {len(predicted)} elements, C has {len(C)}."
        )

    return [
        {
            "c":                c,
            "x":                x,
            "predicted":        predicted[i],
            "ground_truth":     ground_truth[i],
            "correct":          predicted[i] == ground_truth[i],
            "model":            row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "timestamp":        row["timestamp"],
            "parse_successful":   True,
            "status": "ok",
            "raw_response": row["response"]
        }
        for i, c in enumerate(C)
    ]


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_analysis_df(
    log_path:     str = "usage_log.jsonl",
    mapping_path: str = "err_mapping.json",
) -> pd.DataFrame:
    df = load_log(log_path)
    df = df[df["query_type"] == "err_extraction"].reset_index(drop=True)

    if df.empty:
        raise ValueError("No err_extraction entries found in log.")

    rows = []
    for _, row in df.iterrows():
        rows.extend(explode_entry(row, mapping_path=mapping_path))

    return pd.DataFrame(rows)


def compute_stats(
    exploded: pd.DataFrame,
    groupby: list[str] = ["c", "x"],
) -> pd.DataFrame:
    """
    Returns P(correct) for each (c, x) group, along with sample count.
    groupby can be extended e.g. to ["c", "x", "model"] for cross-model comparison.
    """
    return (
        exploded
        .groupby(groupby)["correct"]
        .agg(p_correct="mean", n_samples="count")
        .reset_index()
        .sort_values(["x", "c"])
    )


def compute_x_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the per-condition stats by 'x' to find the 
    mean and median performance for each experiment.
    """
    return (
        stats.groupby("x")["p_correct"]
        .agg(
            mean_p_correct="mean",
            median_p_correct="median",
            n_conditions="count"  # How many 'c' values were in this 'x'
        )
        .reset_index()
    )

# ── Updated Entry point ───────────────────────────────────────────────────────


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    log_path:     str = "usage_log.jsonl",
    mapping_path: str = "err_mapping.json",
):
    # 1. Build the exploded dataframe (raw data)
    exploded = build_analysis_df(log_path=log_path, mapping_path=mapping_path)

    # 2. Compute the detailed stats per (c, x)
    stats = compute_stats(exploded)

    # 3. Compute the summary per x (New)
    x_summary = compute_x_summary(stats)
    # x_summary = x_summary[x_summary['x'] >= 600]

    # 4. Sort x_summary by lowest p_correct mean and median

    x_mean_min = x_summary.sort_values(by="mean_p_correct")
    x_median_min = x_summary.sort_values(by="median_p_correct")

    # 5. Find maximum that is uniquly x % k == 0 for given C

    C = [13, 17, 19]  # 11, 13]
    c_target = 17
    # Directly translates your logic: sum(val % c == 0 for c in C) == 1
    df = x_mean_min
    df_filtered = df[df['x'].apply(lambda val: sum(val % c == 0 for c in C) == 1 and val % c_target == 0)]

    print("\n=== Per (c, x) correctness ===")
    print(stats.to_string(index=False))

    print("\n=== Aggregate Stats Per x ===")
    print(x_summary.to_string(index=False))

    print("\n=== Top 10 Min mean_p_corrt Stats Per x ===")
    print(x_mean_min.head(10).to_string(index=False))

    print("\n=== Min median_p_corrt Stats Per x ===")
    print(x_median_min.head(10).to_string(index=False))

    print("\n=== Valid positive min x_median Stats Per x ===")
    print(df_filtered.sort_values(by='mean_p_correct', ascending=True).head(10).to_string(index=False))

    print("\n=== Valid positive max x_median Stats Per x ===")
    print(df_filtered.sort_values(by='mean_p_correct', ascending=False).head(10).to_string(index=False))

    return exploded, stats, x_summary


if __name__ == "__main__":
    run()
