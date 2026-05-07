import numpy as np
import pandas as pd
import json
import ast
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


def load_log(log_path: str = "usage_log.jsonl") -> pd.DataFrame:
    rows = []
    with Path(log_path).open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_df_induc(path_to_df):
    df = load_log(path_to_df)
    df = df[df["query_type"] == "concept_identification"]

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df['positive'] = [d['positive'] for d in df['metadata']]
    df['negative'] = [d['negative'] for d in df['metadata']]
    df['c_target'] = [d['c_target'] for d in df['metadata']]

    df["incorrect"] = df["parsed_response"] != df['c_target']

    result = df.groupby(["positive", "negative", "c_target"])["incorrect"].mean().reset_index()
    result.rename(columns={"incorrect": "y-axis"}, inplace=True)
    return result


def get_x_axis_for_pairs(df_induc: pd.DataFrame, df_deduc: pd.DataFrame) -> pd.DataFrame:
    deduc_lookup = df_deduc.set_index('x')['x-axis'].to_dict()

    def pair_error(row):
        err_pos = deduc_lookup.get(row['positive'], np.nan)
        err_neg = deduc_lookup.get(row['negative'], np.nan)
        return np.nanmean([err_pos, err_neg])

    df_induc['mean_err'] = df_induc.apply(pair_error, axis=1)
    df_induc['x-axis'] = df_induc['mean_err']

    return df_induc[['positive', 'negative', 'c_target', 'y-axis', 'mean_err', 'x-axis']]


def find_c_target(C, x):
    for c in C:
        if x % c == 0:
            return c


def get_df_deduc_mean(path_to_df):
    df = load_log(path_to_df)

    # query type filter
    query_type = "err_extraction"
    df = df[df["query_type"] == query_type]

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # This handles the extraction safely and is usually faster
    df['x'] = [d.get('x') for d in df['metadata']]
    df['c'] = [d.get('c') for d in df['metadata']]
    df["ground_truth"] = [d.get('ground_truth') for d in df['metadata']]

    df = df[["x", "c", "ground_truth", "parsed_response"]]
    df["correct"] = df["ground_truth"] == df["parsed_response"]

    # 1. Create a column for incorrect answers (True if wrong, False if right)
    df["incorrect"] = ~df["correct"]

    # 2. Group by x and c, then take the mean of the 'incorrect' column
    result = df.groupby(["x"])["incorrect"].mean().reset_index()

    # 3. Rename the column for clarity
    result.rename(columns={"incorrect": "x-axis"}, inplace=True)

    mean_over_x = result.groupby(["x"])["x-axis"].mean().reset_index()
    print(mean_over_x.sort_values("x", ascending=False).head())
    return mean_over_x


def get_df_deduc_mean_weighted(path_to_df):
    df = load_log(path_to_df)
    df = df[df["query_type"] == "err_extraction"]
    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['x'] = [d.get('x') for d in df['metadata']]
    df['c'] = [d.get('c') for d in df['metadata']]
    df["ground_truth"] = [d.get('ground_truth') for d in df['metadata']]
    df = df[["x", "c", "ground_truth", "parsed_response"]]
    df["correct"] = df["ground_truth"] == df["parsed_response"]
    C = df["c"].unique()
    df["incorrect"] = ~df["correct"]
    target_map = {val: find_c_target(x=val, C=C) for val in df['x'].unique()}
    concept_means = df.groupby(['x', 'c'])['incorrect'].mean().reset_index()
    concept_means['is_target'] = concept_means.apply(
        lambda row: row['c'] == target_map[row['x']], axis=1
    )
    pivot_df = concept_means.groupby(['x', 'is_target'])['incorrect'].mean().unstack()
    results = (pivot_df[True].fillna(0) * 0.5) + (pivot_df[False].fillna(0) * 0.5)
    final_df = results.reset_index()
    final_df.columns = ['x', 'x-axis']
    return final_df


def plot(merged: pd.DataFrame):

    n_categories = merged['c_target'].nunique()
    marker_list = ['o', 's', 'D', 'v', '^', '<', '>'][:n_categories]
    sns.set_theme(style='white', font_scale=1.8)  # scale all text up
    sns.set_palette("Set1")  # or "Set1" for even more saturated colors

    fig_width = 10
    fig_height = 10

    g = sns.lmplot(
        data=merged,
        x='x-axis',
        y='y-axis',
        hue='c_target',
        markers=marker_list,  # type: ignore
        ci=None,
        height=fig_height,          # FIX 1: was fig_width
        aspect=fig_width/fig_height,
        scatter_kws={'s': 100, 'alpha': 0.8},
    )

    ax = g.axes[0][0]
    ax.plot([0, 1], [0, 1], color='lightgrey', linewidth=1.5, linestyle='--', zorder=0)

    # Extend each regression line to full x-axis range
    x_min, x_max = 0, 1
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        dx = x[-1] - x[0]
        if dx == 0:                 # FIX 5: guard against zero division
            continue
        slope = (y[-1] - y[0]) / dx
        intercept = y[0] - slope * x[0]
        line.set_xdata([x_min, x_max])
        line.set_ydata([slope * x_min + intercept, slope * x_max + intercept])

    g.set_axis_labels(

        'Mean deductive error',
        'Total teaching error'
    )
    g.set(xlim=(0, 1.05), ylim=(0, 1.05))
    # FIX 2: use suptitle for FacetGrid; FIX 3: removed set_box_aspect
    # g.figure.suptitle('Linear Fit: Teaching Error vs Deductive Error by Category', y=1.01)
    sns.move_legend(
        g,
        loc='center right',
        bbox_to_anchor=(0.85, 0.23),
        frameon=False
    )

    plt.savefig("2td_high_dpi.png", bbox_inches='tight', dpi=600)  # FIX 4: extension + tight bbox
    plt.show()


def run():
    df_induc = get_df_induc("usage_log_2td_concept.jsonl")
    df_deduc = get_df_deduc_mean("usage_log_td2.jsonl")
    merged = get_x_axis_for_pairs(df_induc, df_deduc)

    print(merged.sort_values("x-axis"))
    plot(merged)
