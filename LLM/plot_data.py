import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

import seaborn as sns


def plot_avilable_data(filename: str):
    if not Path(filename).exists():
        raise FileExistsError("No such file found: ", filename)

    df = pd.read_csv(filename)

    # Set aesthetic parameters
    sns.set_theme(style="whitegrid")

    # Create the 3x5 grid
    g = sns.relplot(
        data=df,
        x="x",
        y="p_err",
        hue="k",
        col="model",
        kind="line",
        col_wrap=5,       # Forces the 5-column layout (3 rows if 15 models)
        palette="husl",   # Provides a wide, diverse range of colors
        alpha=0.5,        # Lowers the opacity/intensity of the lines
        marker=None,      # Ensures lines only, no dots
        linewidth=2,      # Slightly thicker lines to keep them visible while transparent
        height=3,
        aspect=1.2
    )

    # Clean up titles and labels
    g.set_titles("{col_name}")
    g.set_axis_labels("X Variable", "Error Probability (p_err)")

    # Optional: adjust spacing to ensure labels don't overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    plt.savefig("model_comparison_plot.png", dpi=300)


def main():
    file_name = "../Results/multi_model/slm_no_think_c=1--15_x=200_benchmark_v3.csv"
    plot_avilable_data(filename=file_name)


if __name__ == "__main__":
    main()
