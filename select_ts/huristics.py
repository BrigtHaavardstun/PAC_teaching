from select_ts.custom_typing import *
from select_ts.ctd import find_min_ts

from typing import List
import pandas as pd
import numpy as np
import math


def select_teaching_set(c_target: int, X: List[int], C: List[int], p: float, q: float, err: GetError, label: GetLabel, sim_mode_L=False, h_mode_u=True, ts_size=5):
    def sim(c: int, c_target: int) -> float:
        sum_prob_same = 0
        for x in X:
            if label(c=c, x=x) == label(c=c_target, x=x):
                sum_prob_same += 1

        return sum_prob_same / len(X)

    def sim_L(c: int, c_target: int) -> float:
        sum_prob_same = 0.0
        for x in X:
            if label(c, x) == label(c_target, x):
                sum_prob_same += 1-err(c=c, x=x)
            else:
                sum_prob_same += err(c=c, x=x)

        return sum_prob_same / len(X)

    def h_adj_uniquness_scores(c_target: int, X: List[int], C: List[int], G: List[int], B: List[int], label: GetLabel, err: GetError):
        EPS = 1e-9
        uniqueness_scores = {x: 0.0 for x in X}
        for x in X:
            c_target_label = label(c=c_target, x=x)
            p_target_correct = 1 - err(c=c_target, x=x)
            if p_target_correct < EPS:          # LLM always wrong on target → useless x
                continue                         # score stays 0.0

            log_prob = 0.0
            degenerate = False
            for c in C:
                if c == c_target:
                    continue
                if label(c=c, x=x) == c_target_label:
                    p = err(c=c, x=x)           # want LLM to err on c
                else:
                    p = 1 - err(c=c, x=x)      # want LLM to be correct on c
                if p < EPS:                     # factor is 0 → x can't distinguish this c
                    degenerate = True
                    break
                log_prob += math.log(p)

            if not degenerate:
                uniqueness_scores[x] = p_target_correct * math.exp(log_prob)

        return uniqueness_scores

    def h_adj_homogeneity(c_target: int, X: List[int], C: List[int], G: List[int], B: List[int], label: GetLabel, err: GetError):
        homogenitiy_scores = {x: 0.0 for x in X}

        for x in X:
            label_c_target = label(c=c_target, x=x)
            expected_nr_good = 0.0
            expected_nr_bad = 0.0

            for g in G:
                if label(c=g, x=x) == label_c_target:
                    expected_nr_good += 1 - err(c=g, x=x)
                else:
                    expected_nr_good += err(c=g, x=x)

            for b in B:
                if label(c=b, x=x) == label_c_target:
                    expected_nr_bad += 1 - err(c=b, x=x)
                else:
                    expected_nr_bad += err(c=b, x=x)

            prob_of_good = (expected_nr_good) / (expected_nr_good + expected_nr_bad)
            homogenitiy_scores[x] = prob_of_good  # we minimize the prob of bad!

        return homogenitiy_scores

    sim_curr: SimFunc = sim_L if sim_mode_L else sim
    hurristic_curr: HuristicFunc = h_adj_uniquness_scores if h_mode_u else h_adj_homogeneity
    # Make a set of ease for quick check of inclusion
    C_set = set(C)
    G_set = {c for c in C if sim_curr(c=c, c_target=c_target) >= q}
    B_set = C_set - G_set

    G = sorted(G_set)
    B = sorted(B_set)

    h_scores = hurristic_curr(c_target=c_target, X=X, C=C, G=G, B=B, label=label, err=err)
    best_x = sorted(X, key=lambda x: h_scores[x], reverse=True)
    best_x_0 = [x for x in best_x if not label(c=c_target, x=x)]
    best_x_1 = [x for x in best_x if label(c=c_target, x=x)]
    selected_x = []
    current = best_x_0
    other = best_x_1
    while len(selected_x) < ts_size:
        if not current and not other:
            break
        if not current:
            current, other = other, current
        val = current.pop(0)
        current, other = other, current
        selected_x.append(val)

    selected_teaching_set = [(x, label(c=c_target, x=x)) for x in selected_x]
    return selected_teaching_set


def get_error(slm: str) -> GetError:
    df = pd.read_csv("Results/LLM_response_x_mult_k.csv")
    df = df[df["model"] == slm]

    # Oppretter en dictionary med (k, x) som nøkkel og p_err som verdi
    # Dette gir O(1) oppslagstid
    error_dict = dict(zip(zip(df['k'], df['x']), df['p_err']))

    def get_err(c: int, x: int) -> float:
        # Bruker .get() for å unngå krasj hvis kombinasjonen ikke finnes
        # Returnerer 0.0 (eller en annen default) hvis (c, x) mangler
        return error_dict[(c, x)]
    return get_err


def get_no_error(c: int, x: int) -> float:
    return 0.0


def get_mult_k_label(c: int, x: int) -> bool:
    return x % c == 0


def escape_latex(text: str) -> str:
    """Escapes common LaTeX special characters."""
    return text.replace("_", "\\_").replace("/", "/\\allowbreak ")


def run():
    SLMS = [
        "TD",
        "no_error",
        # "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen3-4B-instruct-2507",
        # "google/gemma-4-e2b-it",
        # "microsoft/Phi-4-mini-instruct"
    ]

    C = [13, 23, 29]
    C_target = C
    X = list(range(350, 400))
    settings = [(False, True), (False, True), (True, False), (True, True)]

    for sim, h in settings:

        # --- Dynamic Caption Logic ---
        # Map True/False to your specific LaTeX strings
        h_text = "H\\_uniqueness" if h else "H\\_homogeneity"
        sim_text = "sim-L" if sim else "sim"

        full_caption = (
            f"Teaching sets generated for various error sources and target concepts. "
            f"Using {h_text} and {sim_text}. "
            f"Four of the error sources are the LLMs, while the "
            f"'no\\_error' model serves as a minimum TD teaching set as a baseline comparrision ."
        )

        # Buffer to hold our lines
        rows = []

        # 1. Build the Table Structure
        rows.append("\\begin{table}[h]")
        rows.append("\\centering")
        rows.append("\\begin{tabular}{llccc}")
        rows.append("\\toprule")
        rows.append("Target & Error source & Item 1 & Item 2 & Item 3 \\\\")
        rows.append("\\midrule")

        # 2. Logic Loop
        for c in C_target:
            num_models = len(SLMS)
            for i, slm in enumerate(SLMS):
                if slm == "TD":
                    ts = sorted(find_min_ts(c))
                else:
                    err = get_no_error if slm == "no_error" else get_error(slm=slm)
                    label = get_mult_k_label
                    ts = sorted(select_teaching_set(
                        c_target=c, X=X, C=C, p=0.75, q=0.95,
                        err=err, label=label, sim_mode_L=sim, h_mode_u=h, ts_size=3
                    ))

                items = [f"({val}, {'T' if lbl else 'F'})" for val, lbl in ts]
                while len(items) < 3:
                    items.append("-")

                target_col = f"\\multirow{{{num_models}}}{{*}}{{{c}}}" if i == 0 else ""
                model_name = escape_latex(slm)

                # Add formatted row to list
                rows.append(f"{target_col} & {model_name} & {items[0]} & {items[1]} & {items[2]} \\\\")

            rows.append("\\midrule")

        # 3. Footer
        rows.append("\\bottomrule")
        rows.append("\\end{tabular}")
        rows.append(f"\\caption{{{full_caption}}}")
        rows.append("\\end{table}")

        # 4. Final Write: Join everything with newlines
        sim_type = 'sim-L' if sim else 'sim'
        h_type = 'h-u' if h else 'h-h'
        filename = f"Results/Huristics/Tabels/teaching_set_{h_type}_{sim_type}.tex"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))

        print(f"Generated {filename}")


if __name__ == "__main__":
    run()
