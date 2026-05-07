import ast
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from select_ts.huristics import (
    get_good_bad_concepts,
    get_mult_k_label,
    h_approix,
    h_adj_homogeneity,
    h_adj_uniquness_scores,
)

from OPENAI.new_analysis import (get_df_deduc_mean as td1_deduc, get_df_induc as td1_induc)
from OPENAI.new_analysis_2td import (get_df_deduc_mean as td2_deduc, get_df_induc as td2_induc)


CONCEPTS = [5, 7, 11, 13, 17]
Q_THRESHOLD = 0.95
RESULTS_DIR = Path("Results")


def load_log(log_path: str) -> pd.DataFrame:
    rows = []
    with Path(log_path).open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_metadata(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def build_error_lookup(log_path: str) -> dict[tuple[int, int], float]:
    df = load_log(log_path)
    df = df[df["query_type"] == "err_extraction"].copy()
    df["metadata"] = df["metadata"].apply(normalize_metadata)
    df["x"] = [int(d.get("x")) for d in df["metadata"]]
    df["c"] = [int(d.get("c")) for d in df["metadata"]]
    df["ground_truth"] = [d.get("ground_truth") for d in df["metadata"]]
    df["incorrect"] = df["ground_truth"] != df["parsed_response"]

    grouped = df.groupby(["c", "x"])["incorrect"].mean()
    return {(int(c), int(x)): float(p_err) for (c, x), p_err in grouped.items()}


def print_latex_table(C, selected_uniquess, selected_homoginity, size=1):
    """
    selected_uniquess[c_target]  = (x, score, deduc_mean_err, inductive_err)
    selected_homoginity[c_target] = (x, score, deduc_mean_err, inductive_err)
    """
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Heuristics for teaching set size " + str(size) + "}")

    print(r"\begin{tabular}{clcccc}")
    print(r"\hline")
    print(r"$c_{\text{target}}$ & Criterion & $x$ & Heuristic Score & Mean Deductive Error & Teaching Error \\")
    print(r"\hline")

    for c_target in C:
        uniquess_x, uniquess_x_score, deduc_mean_err_uniq, inductive_err_uniq = selected_uniquess[c_target]
        homoginity_x, homoginity_x_score, deduc_mean_err_hom, inductive_mean_err_hom = selected_homoginity[c_target]

        print(f"\\multirow{{2}}{{*}}{{{c_target}}}")
        print(
            f"    & Uniqueness  & {uniquess_x}  & {round(uniquess_x_score, 3)} & {deduc_mean_err_uniq:.2f} & {inductive_err_uniq:.2f} \\\\")
        print(
            f"    & Homogeneity & {homoginity_x} & {round(homoginity_x_score, 3)} & {deduc_mean_err_hom:.2f} & {inductive_mean_err_hom:.2f} \\\\")
        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\label{tab:results}")
    print(r"\end{table}")


def gather_data_td1(C: list[int]):
    df_induc = td1_induc("usage_log_100.jsonl")
    df_deduc = td1_deduc("usage_log_100.jsonl")

    all_x = list(map(int, df_deduc["x"].unique()))
    selected_unqiuess = {}
    selected_homoginity = {}
    err_lookup = build_error_lookup("usage_log_100.jsonl")

    def get_err(c, x):
        result = err_lookup[(c, x)]
        return result

    for c_target in C:

        score_unqiueness = h_adj_uniquness_scores(c_target=c_target, X=all_x, C=C, label=get_mult_k_label, err=get_err)
        # minimize, want few expect matches
        best_unqiueness = sorted(score_unqiueness.items(), key=lambda x: (x[1], x[0]), reverse=False)

        score_homogenitiy = h_adj_homogeneity(c_target=c_target, X=all_x, C=C, label=get_mult_k_label, err=get_err)
        # maximize, want lots of similar concepts
        best_homogenitiy = sorted(score_homogenitiy.items(), key=lambda x: (x[1], -x[0]), reverse=True)

        x = best_unqiueness[0][0]
        score = best_unqiueness[0][1]
        selected_unqiuess[c_target] = (x, score, float(df_deduc.loc[df_deduc['x'] == x, 'x-axis'].iloc[0]),
                                       float(df_induc.loc[df_induc['x'] == x, 'y-axis'].iloc[0]))

        x = best_homogenitiy[0][0]
        score = best_homogenitiy[0][1]
        selected_homoginity[c_target] = (x, score, float(df_deduc.loc[df_deduc['x'] == x, 'x-axis'].iloc[0]),
                                         float(df_induc.loc[df_induc['x'] == x, 'y-axis'].iloc[0]))

    for c_target in C:
        uniquess_x = selected_unqiuess[c_target][0]
        uniquess_x_score = selected_unqiuess[c_target][1]
        # print(uniquess_x)
        deduc_mean_err_uniq = float(df_deduc.loc[df_deduc['x'] == uniquess_x, 'x-axis'].iloc[0])
        # print(deduc_mean_err)
        inductive_err_uniq = float(df_induc.loc[df_induc['x'] == uniquess_x, 'y-axis'].iloc[0])

        homoginity_x = selected_homoginity[c_target][0]
        homoginity_x_score = selected_homoginity[c_target][1]
        deduc_mean_err_hom = float(df_deduc.loc[df_deduc['x'] == homoginity_x, 'x-axis'].iloc[0])
        inductive_mean_err_hom = float(df_deduc.loc[df_deduc['x'] == homoginity_x, 'x-axis'].iloc[0])

        print(f"{c_target=} {uniquess_x=}, {uniquess_x_score=}, {deduc_mean_err_uniq=}, {inductive_err_uniq=}")
        print(f"{c_target=} {homoginity_x=},{homoginity_x_score=} {deduc_mean_err_hom=}, {inductive_mean_err_hom=}")

    print_latex_table(C=C, selected_uniquess=selected_unqiuess,
                      selected_homoginity=selected_homoginity)


def gather_data_td2(C: list[int]):
    df_induc = td2_induc("usage_log_2td_concept.jsonl")
    df_deduc = td2_deduc("usage_log_td2.jsonl")

    all_x = list(map(int, df_deduc['x'].unique()))

    selected_unqiuess = {}
    selected_homoginity = {}
    err_lookup = build_error_lookup("usage_log_td2.jsonl")

    def get_err(c, x):
        result = err_lookup[(c, x)]
        return result

    for c_target in C:
        df_c = df_induc[df_induc["c_target"] == c_target]
        all_pairs = list(df_c[['positive', 'negative']].drop_duplicates().itertuples(index=False, name=None))

        score_unqiueness = h_adj_uniquness_scores(c_target=c_target, X=all_x, C=C, label=get_mult_k_label, err=get_err)
        # minimize, want few expect matches
        pair_score_uniquness = {}
        for pair in all_pairs:
            pair_score_uniquness[pair] = (score_unqiueness[pair[0]] + score_unqiueness[pair[1]])/2

        best_unqiueness_pair = sorted(pair_score_uniquness.items(), key=lambda x: (x[1], x[0][0]), reverse=False)

        score_homogenitiy = h_adj_homogeneity(c_target=c_target, X=all_x, C=C, label=get_mult_k_label, err=get_err)
        # maximize, want lots of similar concepts
        pair_score_hom = {}
        for pair in all_pairs:
            pair_score_hom[pair] = (score_homogenitiy[pair[0]] + score_homogenitiy[pair[1]])/2

        best_hom_pair = sorted(pair_score_hom.items(), key=lambda x: (x[1], -x[0][0]), reverse=True)

        pos, neg = best_unqiueness_pair[0][0]
        score = best_unqiueness_pair[0][1]
        deduc_mean_err_pos = float(df_deduc.loc[df_deduc['x'] == pos, 'x-axis'].iloc[0])
        deduc_mean_err_negative = float(df_deduc.loc[df_deduc['x'] == neg, 'x-axis'].iloc[0])
        deduc_mean_err = (deduc_mean_err_pos+deduc_mean_err_negative) / 2

        inductive_err = float(df_induc.loc[(df_induc['positive'] == pos) &
                              (df_induc['negative'] == neg), 'y-axis'].iloc[0])

        print(deduc_mean_err, inductive_err)

        selected_unqiuess[c_target] = ((pos, neg), score, deduc_mean_err, inductive_err)

        pos, neg = best_hom_pair[0][0]
        score = best_unqiueness_pair[0][1]
        deduc_mean_err_pos = float(df_deduc.loc[df_deduc['x'] == pos, 'x-axis'].iloc[0])
        deduc_mean_err_negative = float(df_deduc.loc[df_deduc['x'] == neg, 'x-axis'].iloc[0])
        deduc_mean_err = (deduc_mean_err_pos+deduc_mean_err_negative) / 2

        inductive_err = float(df_induc.loc[(df_induc['positive'] == pos) &
                              (df_induc['negative'] == neg), 'y-axis'].iloc[0])
        selected_homoginity[c_target] = ((pos, neg), score, deduc_mean_err, inductive_err)

    print_latex_table(C=C, selected_uniquess=selected_unqiuess, selected_homoginity=selected_homoginity)


def get_mult_k_label(c: int, x: int) -> bool:
    return x % c == 0


def run() -> pd.DataFrame:
    C = [5, 7, 11, 13, 17]
    # gather_data_td1(C=C)
    gather_data_td2(C=C)


if __name__ == "__main__":
    run()


# lag en tabell for alle. opggi heuristic score, over all score 3/20 for y verdi, x verdi og hvilket teaching set det er
# Oppdater for 2td
