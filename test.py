
from utils.cache import get_errors
from select_ts.find_high_err_td import find_td_for_concepts
from experiment.LLM_version_space_learner import run
from itertools import combinations
from select_ts.huristics import run as run_huristic
from experiment.generation_evaluation import run as sanity_check
# from experiment.gen_err import run as gen_err_run


def find_hard_concept():
    X = list(range(350, 400))
    C = list(range(10, 30))
    for comb in combinations(C, 3):
        valid, ts, c = find_td_for_concepts(X=X, C=list(comb))
        if valid:
            print("Golden comb:", comb, "target:", c, "ts:",  ts)


def inference():
    concepts = [574, 352, 731]

    test_cases_ps_ns_k = [
        # Easy: k=11, negatives rule out all neighbours
        ([125706], [107931, 70048], 574),
        ([403512], [365201, 202048], 731)

    ]

    model_id = "google/gemma-4-e2b-it"

    answer = run(model_id=model_id, test_cases_ps_ns_k=test_cases_ps_ns_k, concepts=concepts)
    answer.to_csv("test.csv")
    print(answer)


def calc_errs():
    C = list(range(573, 30))
    X = list(range(350, 400))

    model_id = "Qwen/Qwen3-4B-instruct-2507"
    pairs = [(k, x) for k in C for x in X]
    all_errors = get_errors(model_id=model_id, pairs=pairs)
    for key, value in all_errors.items():
        print(f"k={key[1]}, x={key[2]}, p_err={round(value, 4)}")


def add_to_json():
    import json

    # 1. Load the existing data
    with open('err_mapping.json', 'r') as f:
        current_data = json.load(f)

    # Define your data in a dictionary
    data = """T
F
F
But must output comma-separated on single line with T/F values in order. So: 
T,F,F"""
    current_data[data] = [False, False, False, False, False, False]
    # Write to file
    with open('err_mapping.json', 'w') as f:
        json.dump(current_data, f, indent=4)


def random_plot():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Setup parameters
    # x-axis: Probability of success
    probabilities = [1, 0.9, 0.7, 0.5, 0.4, 0.3, 0.1, 0.001]
    # Each x value has 10 experiments
    n_experiments_per_p = 10
    # Each experiment consists of 10 trials
    trials_per_experiment = 10

    # 2. Simulate data
    data = []
    for p in probabilities:
        # Use the binomial distribution to simulate 'y' (number of successes)
        # n=10 trials, probability=p, repeated 10 times for this specific x
        y_values = np.random.binomial(trials_per_experiment, p, n_experiments_per_p)
        for y in y_values:
            data.append({'Probability': p, 'Successes': y})

    df = pd.DataFrame(data)

    # 3. Create Visualization
    plt.figure(figsize=(10, 6))

    # Boxplot shows the quartiles and median of the 10 experiments
    sns.boxplot(x='Probability', y='Successes', data=df, color='lightgray', showfliers=False)

    # Stripplot overlays the 10 individual data points (jitter helps avoid overlap)
    sns.stripplot(x='Probability', y='Successes', data=df, jitter=True,
                  size=8, palette="viridis", hue='Probability', legend=False)

    # Formatting
    plt.title('Distribution of Successes ($y$) for each Probability ($x$)', fontsize=14)
    plt.xlabel('Probability of Success ($p$)', fontsize=12)
    plt.ylabel('Number of Successes (out of 10)', fontsize=12)
    plt.ylim(-0.5, 10.5)
    plt.yticks(range(11))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Save and show
    plt.savefig('success_distribution.png')


if __name__ == "__main__":
    # find_hard_concept()
    # calc_errs()
    # sanity_check()
    # inference()
    # gen_err_run()
    # run_huristic()
    # add_to_json()
    random_plot()
