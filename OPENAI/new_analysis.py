import numpy as np
import pandas as pd
import json
from pathlib import Path
import ast

import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde


def load_log(log_path: str = "usage_log.jsonl") -> pd.DataFrame:
    rows = []
    with Path(log_path).open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_df_induc(path_to_df):

    df = load_log(path_to_df)

    # query type filter
    query_type = "concept_identification"
    df = df[df["query_type"] == query_type]

    # Time filter
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # before = pd.to_datetime('2026-04-29 13:50:00', utc=True)
    # after = pd.to_datetime('2026-04-29 13:00:00', utc=True)

    # filtered_df = df[(df['timestamp'] < before) & (after < df['timestamp'])]
    # df = filtered_df

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # This handles the extraction safely and is usually faster
    df['x'] = [d['x'] for d in df['metadata']]
    df['c_target'] = [d['c_target'] for d in df['metadata']]
    df["correct"] = df["parsed_response"] == df['c_target']

    df["incorrect"] = ~df["correct"]

    result = df.groupby(["x", "c_target"])["incorrect"].mean().reset_index()

    result.rename(columns={"incorrect": "y-axis"}, inplace=True)

    # print(result.sort_values("y-axis", ascending=False).head())

    return result


def get_df_induc_mult(path_to_df):
    df = load_log(path_to_df)

    # query type filter
    query_type = "concept_identification"
    df = df[df["query_type"] == query_type]

    # Time filter
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # before = pd.to_datetime('2026-04-29 13:50:00', utc=True)
    # after = pd.to_datetime('2026-04-29 13:00:00', utc=True)

    # filtered_df = df[(df['timestamp'] < before) & (after < df['timestamp'])]
    # df = filtered_df

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # This handles the extraction safely and is usually faster
    df['x'] = [d.get('x') for d in df['metadata']]
    df['c_target'] = [d.get('c_target') for d in df['metadata']]

    print(df["parsed_response"])
    df["parsed_response"] = df.apply(lambda row: [] if row["parsed_response"] ==
                                     # df["list_response"] = df.apply(lambda row: ast.literal_eval(row["parsed_response"]), axis=1)
                                     "N/A" else row["parsed_response"], axis=1)
    print(df["parsed_response"])
    df["correct"] = df.apply(lambda row: True if (len(row["parsed_response"]) == 1) and (int(row[
        "parsed_response"][0]) == int(row['c_target'])) else False, axis=1)

    result = df.groupby(["x", "c_target"])["correct"].mean().reset_index()
    result.rename(columns={"correct": "y-axis"}, inplace=True)

    print(result.sort_values("y-axis").head())
    print(result.sort_values("y-axis", ascending=False).head())

    return result


def find_c_target(C, x):
    for c in C:
        if x % c == 0:
            return c


def filter_on_c_target_accuracy(df, acc_threshold: float):
    perfect_x = []
    all_c = list(set(df["c"]))
    for x in df['x'].unique():
        df_x = df[df["x"] == x]
        c_target = find_c_target(C=all_c, x=x)
        df_x_c = df_x[df_x["c"].astype(int) == c_target]
        mean_accuracy = df_x_c["correct"].mean()
        print(mean_accuracy)
        if mean_accuracy >= acc_threshold:
            perfect_x.append(x)
    df = df[~df["x"].isin(perfect_x)]
    return df


def get_df_deduc(path_to_df, acc_threshold=0.9):

    df = load_log(path_to_df)

    query_type = "err_extraction"
    df = df[df["query_type"] == query_type]

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["metadata"] = df["metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df["x"] = [d.get("x") for d in df["metadata"]]
    df["c"] = [d.get("c") for d in df["metadata"]]
    df["ground_truth"] = [d.get("ground_truth") for d in df["metadata"]]

    df = df[["x", "c", "ground_truth", "parsed_response"]]
    df["correct"] = df["ground_truth"] == df["parsed_response"]
    # filter
    df = filter_on_c_target_accuracy(df, acc_threshold=acc_threshold)
    # Per (x, c): mean probability of being correct
    per_xc = df.groupby(["x", "c"])["correct"].mean().reset_index()
    per_xc.rename(columns={"correct": "p_correct"}, inplace=True)

    per_xc["is_target"] = per_xc.apply(
        lambda r: r["x"] % r["c"] == 0,
        axis=1
    )
    # Remove all x where target concept has p_correct == 1
    bad_x = per_xc.loc[
        (per_xc["is_target"]) & (per_xc["p_correct"] == 1),
        "x"
    ]
    # per_xc = per_xc[~per_xc["x"].isin(bad_x)]
    # ---------- numerically stable helpers ----------

    def safe_log(p):
        if p <= 0.0:
            return -np.inf
        return np.log(p)

    def pac_probability(group):
        target = group[group["is_target"]]
        others = group[~group["is_target"]]

        p_target = target["p_correct"].values[0]
        q_others = (1 - others["p_correct"]).values

        log_p_target = safe_log(p_target)
        log_fail_target = safe_log(1 - p_target)

        log_q_others = np.array([safe_log(q) for q in q_others])
        log_not_q_others = np.array(
            [safe_log(1 - q) for q in q_others]
        )

        n = len(q_others)
        total_concepts = n + 1

        log_result = -np.inf

        # Case 1: target succeeds, k others succeed
        for k in range(n + 1):

            log_weight = safe_log(1.0 / (k + 1))

            for subset in combinations(range(n), k):

                subset_set = set(subset)

                log_p_subset = 0.0

                for i in range(n):
                    if i in subset_set:
                        log_p_subset += log_q_others[i]
                    else:
                        log_p_subset += log_not_q_others[i]

                log_term = (
                    log_p_target
                    + log_p_subset
                    + log_weight
                )

                log_result = np.logaddexp(
                    log_result,
                    log_term
                )

        # Case 2: target fails AND all others fail

        log_all_others_fail = np.sum(log_not_q_others)

        log_term = (
            log_fail_target
            + log_all_others_fail
            + safe_log(1.0 / total_concepts)
        )

        log_result = np.logaddexp(
            log_result,
            log_term
        )

        return float(np.exp(log_result))

    # ---------- final steps preserved exactly ----------

    result = (
        per_xc
        .groupby("x")
        .apply(pac_probability)
        .reset_index()
    )

    result.columns = ["x", "x-axis"]

    print(
        result
        .sort_values("x-axis", ascending=False)
        .head()
    )

    return result


def get_df_deduc_mean(path_to_df):
    df = load_log(path_to_df)

    # query type filter
    query_type = "err_extraction"
    df = df[df["query_type"] == query_type]

    # Time filter
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # before = pd.to_datetime('2026-04-29 13:50:00', utc=True)
    # after = pd.to_datetime('2026-04-29 13:00:00', utc=True)

    # filtered_df = df[(df['timestamp'] < before) & (after < df['timestamp'])]
    # df = filtered_df

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

    # query type filter
    query_type = "err_extraction"
    df = df[df["query_type"] == query_type]

    # Time filter
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # before = pd.to_datetime('2026-04-29 13:50:00', utc=True)
    # after = pd.to_datetime('2026-04-29 13:00:00', utc=True)

    # filtered_df = df[(df['timestamp'] < before) & (after < df['timestamp'])]
    # df = filtered_df

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # This handles the extraction safely and is usually faster
    df['x'] = [d.get('x') for d in df['metadata']]
    df['c'] = [d.get('c') for d in df['metadata']]
    df["ground_truth"] = [d.get('ground_truth') for d in df['metadata']]

    df = df[["x", "c", "ground_truth", "parsed_response"]]
    df["correct"] = df["ground_truth"] == df["parsed_response"]
    C = df["c"].unique()
    df['is_target'] = df.apply(lambda row: row['c'] == find_c_target(x=row['x'], C=C), axis=1)
    # 1. Create a column for incorrect answers (True if wrong, False if right)
    df["incorrect"] = ~df["correct"]

    # 1. Map the targets to a dictionary for speed
    target_map = {val: find_c_target(x=val, C=C) for val in df['x'].unique()}

    # 2. Pre-calculate the mean 'incorrect' for each unique (x, c) pair
    # This prevents specific concepts with more samples from overpowering others
    concept_means = df.groupby(['x', 'c'])['incorrect'].mean().reset_index()

    # 3. Label whether each row in our grouped data is the target
    concept_means['is_target'] = concept_means.apply(
        lambda row: row['c'] == target_map[row['x']], axis=1
    )

    # 4. Group by 'x' and 'is_target', then pivot to get Target vs. Others
    pivot_df = concept_means.groupby(['x', 'is_target'])['incorrect'].mean().unstack()

    # 5. Calculate weighted average and format the final DataFrame
    # We use .fillna() to ensure that if an 'x' is missing one type, it doesn't break
    results = (pivot_df[True].fillna(0) * 0.5) + (pivot_df[False].fillna(0) * 0.5)

    # Convert from Series to the requested DataFrame format
    final_df = results.reset_index()
    final_df.columns = ['x', 'x-axis']

    return final_df


def get_df_deduc_err_of_correct(path_to_df):
    df = load_log(path_to_df)

    query_type = "err_extraction"
    df = df[df["query_type"] == query_type]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df['x'] = [d.get('x') for d in df['metadata']]
    df['c'] = [d.get('c') for d in df['metadata']]
    df["ground_truth"] = [d.get('ground_truth') for d in df['metadata']]

    df = df[["x", "c", "ground_truth", "parsed_response"]]
    df["correct"] = df["ground_truth"] == df["parsed_response"]
    df["incorrect"] = ~df["correct"]

    # Filter to only the target concept
    df = df[df.apply(lambda r: r["x"] % r["c"] == 0, axis=1)]

    result = df.groupby(["x"])["correct"].mean().reset_index()
    result.rename(columns={"correct": "x-axis"}, inplace=True)

    print(result.sort_values("x-axis", ascending=False).head())
    return result


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
    g.set(xlim=(0, 1), ylim=(0, 1.05))
    # FIX 2: use suptitle for FacetGrid; FIX 3: removed set_box_aspect
    # g.figure.suptitle('Linear Fit: Teaching Error vs Deductive Error by Category', y=1.01)
    sns.move_legend(
        g,
        loc='center right',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False
    )
    g.legend.remove()

    plt.savefig("1td_high_dpi.png",   bbox_inches='tight', dpi=600)  # FIX 4: extension + tight bbox
    plt.show()


def plot_heatmap(merged: pd.DataFrame, file_name="heat_map"):
    fig, ax = plt.subplots(figsize=(7.2, 6))

    # --- Heatmap of all points combined ---
    x = merged['x-axis'].values
    y = merged['y-axis'].values

    # 2D KDE over the unit square
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    grid_size = 200
    xi = np.linspace(0, 1, grid_size)
    yi = np.linspace(0, 1, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    heatmap = ax.imshow(
        Zi,
        origin='lower',
        extent=[0, 1, 0, 1],
        aspect='auto',
        cmap='YlOrRd',
        interpolation='bilinear',
    )
    fig.colorbar(heatmap, ax=ax, label='Point density')

    # --- Regression line per category ---
    categories = merged['c_target'].unique()
    palette = plt.cm.tab10.colors  # one color per category
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    for i, cat in enumerate(sorted(categories)):
        subset = merged[merged['c_target'] == cat]
        cx = subset['x-axis'].values
        cy = subset['y-axis'].values

        # Fit a line
        slope, intercept = np.polyfit(cx, cy, 1)
        x0, x1 = 0, 1
        color = palette[i % len(palette)]
        ls = linestyles[i % len(linestyles)]

        ax.plot(
            [x0, x1],
            [slope * x0 + intercept, slope * x1 + intercept],
            color=color,
            linestyle=ls,
            linewidth=2,
            label=str(cat),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Probability of all items correct by err <-> simulated PAC teaching')
    ax.set_ylabel('Measured PAC teaching for q=1')
    ax.set_title(f'Linear Fit: Teaching Error vs Deductive Error by Category Acc:{file_name.split("_")[-1]}')
    ax.legend(title='Category', loc='best')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def run():

    deduc_csv_file = "usage_log_100.jsonl"
    df_deduc = get_df_deduc_mean(deduc_csv_file)
    induc_dec_csv = "usage_log_100.jsonl"
    df_induc = get_df_induc(induc_dec_csv)

    merged = df_deduc.merge(df_induc, on="x")

    plot(merged)


if __name__ == "__main__":
    run()


# Endre Plot: Total teaching error
# Endre plot: Mean deductive error
# Fjerne titel


# Legge til en diagonal i grått.


# Endre system prompt til Overleaf stil
# Endre valg av negativt eksempel
