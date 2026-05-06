from typing import List
import pandas as pd
import numpy as np
from itertools import combinations
from utils.cache import get_errors
from collections import defaultdict


def pred(x: int, k: int):
    return x % k == 0


def find_min_td(c_target: int, X: List[int], C: List[int]):
    assert all(a <= b for a, b in zip(X, X[1:]))
    for k in range(1, 5):
        for comb in combinations(X, k):
            ts_target = np.array([pred(x=x, k=c_target) for x in comb])
            valid = True
            for c in C:
                if c == c_target:
                    continue

                ts_concept = np.array([pred(x=x, k=c) for x in comb])
                if np.all(ts_target == ts_concept):
                    valid = False
                    break
            if valid:
                return [(x, b) for x, b in zip(comb, ts_target)]
    raise Exception("No valid Teaching Set found")


def find_td_for_concepts(X: List[int], C: List[int]):
    df = pd.DataFrame()
    c_to_ts = {}
    all_x_selected = []
    x_to_c = defaultdict(list)
    for c in C:
        ts = find_min_td(c_target=c, X=X, C=C)
        c_to_ts[c] = [x for (x, b) in ts]
        for x, b in ts:
            x_to_c[x].append(c)
            all_x_selected.append(x)

    all_pairs = [(c, x) for x in all_x_selected for c in C]
    eval_error_of_all = get_errors(model_id="Qwen/Qwen3-4B-instruct-2507", pairs=all_pairs)
    bad_pairs = []
    for key, val in eval_error_of_all.items():
        if val >= 0.95:
            bad_pairs.append((key[1], key[2]))
            # print(key[1], key[2])
    for c, x in bad_pairs:
        if x in c_to_ts[c]:
            return True, c_to_ts[c], c

    return False, [], None
    # eval_error_of_ts = get_errors(model_id="Qwen/Qwen3-4B-instruct-2507", pairs=all_pairs)
