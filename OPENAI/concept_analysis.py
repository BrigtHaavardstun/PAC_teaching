import json
import pandas as pd
from pathlib import Path
from OPENAI.parsers import parse_concept_identification, parse_with_mapping


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_log(log_path: str = "usage_log.jsonl") -> pd.DataFrame:
    rows = []
    with Path(log_path).open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ── Row builder ───────────────────────────────────────────────────────────────

def build_row(row: pd.Series, mapping_path: str = "concept_mapping.json") -> dict:
    metadata = row["metadata"]
    c_target = metadata["c_target"]
    positive = metadata["positive"]
    negative = metadata["negative"]
    C = metadata["C"]

    sucess, predicted = parse_with_mapping(
        text=row["response"],
        parse_fn=parse_concept_identification,
        mapping_path=mapping_path,
    )
    if not sucess:
        return {
            "c_target":         c_target,
            "positive":         str(positive),   # str for groupby compatibility
            "negative":         str(negative),
            "C":                str(C),
            "predicted":        "NA",
            "correct":          False,
            "model":            row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "timestamp":        row["timestamp"],
            "parse_succesful": False
        }

    return {
        "c_target":         c_target,
        "positive":         str(positive),   # str for groupby compatibility
        "negative":         str(negative),
        "C":                str(C),
        "predicted":        predicted,
        "correct":          predicted == c_target,
        "model":            row["model"],
        "reasoning_effort": row["reasoning_effort"],
        "timestamp":        row["timestamp"],
        "parse_succesful": True

    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_analysis_df(
    log_path:     str = "usage_log.jsonl",
    mapping_path: str = "concept_mapping.json",
) -> pd.DataFrame:
    df = load_log(log_path)
    df = df[df["query_type"] == "concept_identification"].reset_index(drop=True)

    if df.empty:
        raise ValueError("No concept_identification entries found in log.")

    rows = []
    for _, row in df.iterrows():
        rows.append(build_row(row, mapping_path=mapping_path))

    return pd.DataFrame(rows)


def compute_stats(
    df: pd.DataFrame,
    groupby: list[str] = ["c_target", "positive", "negative"],
) -> pd.DataFrame:
    """
    Returns P(correct) for each experiment grouping, along with sample count.
    Default groupby is (c_target, positive, negative) — one row per unique
    experiment setup. Extend to ["c_target", "positive", "negative", "model"]
    for cross-model comparison.
    """
    return (
        df
        .groupby(groupby)["correct"]
        .agg(p_correct="mean", n_samples="count")
        .reset_index()
        .sort_values(["c_target", "p_correct"], ascending=[True, False])
    )


def analysis(df: pd.DataFrame):
    # 1. Filter only on c and positive example
    C = [13, 17, 19]
    X = list(range(400, 600))
    keep_pairs = []
    for c_target in C:
        for x in X:
            if (x % c_target == 0) and (sum(x % c == 0 for c in C) == 1):
                keep_pairs.append((c_target, str([x])))

    filter_df = pd.DataFrame(keep_pairs, columns=['c_target', 'positive'])
    print(filter_df.head(10))
    df_filtered = df.merge(filter_df, on=['c_target', 'positive'])

    # Calculates the mean of 'score' for every group
    print("mean")
    result = df.groupby(['c_target', 'positive'])['correct'].mean().reset_index()
    print(result)

    return result


def run(
    log_path:     str = "usage_log.jsonl",
    mapping_path: str = "concept_mapping.json",
):
    print("ran")
    df = build_analysis_df(log_path=log_path, mapping_path=mapping_path)

    mean_correct_for_c_and_x = analysis(df)

    return df, mean_correct_for_c_and_x


if __name__ == "__main__":
    run()
