
import numpy as np
import pandas as pd
import json
import random
C = [5, 7, 11, 13, 17]
X = list(range(1, 1001))

at_least_one_match = [x for x in X if sum(x % c == 0 for c in C) >= 2]

experiment_ts = []
for x in at_least_one_match:
    matches = set([c for c in C if x % c == 0])
    for c_target in matches:
        rest = matches - {c_target}
        mult_val = list(rest)
        total = 1
        for val in mult_val:
            total *= val

        # bare bruk choice: total

        max_mult = 1000//total
        min_mult = 2

        choices = [total * k for k in range(min_mult, max_mult+1)]
        # Filter away any mults of c_target
        choices = [choice for choice in choices if choice % c_target != 0]

        # Check that at least one other (aside from c_target) has this as a negative example
        choices_2 = [choice for choice in choices if sum(c % choice != 0 for c in set(C)-{c_target})]

        choice = random.choice(choices)

        minimal = random.random() < 0.5
        if minimal:
            choice = total

        print(f"{x=} {c_target=} {choice=} {rest=}")

        experiment_ts.append({
            "c_target": c_target,
            "positive": x,
            "negative": choice
        })


print(experiment_ts)


df = pd.DataFrame(experiment_ts)
df.to_csv("Exerpiment_TS.csv", index=False)

with open("Exerpiment_TS.json", "w") as f:
    json.dump(experiment_ts, f)


unique_x = set()
for row in experiment_ts:
    unique_x.add(row["positive"])
    unique_x.add(row["negative"])
