from OPENAI.concept_analysis import run as run_concept
from OPENAI.err_analysis import run as run_err

import ast


def run():
    _, df_concept = run_concept()
    df_concept["x"] = df_concept["positive"].apply(lambda x: int(ast.literal_eval(x)[0]))
    df_concept["accuracy"] = df_concept["correct"]
    _, _, df_err = run_err()

    print(df_concept.head())
    print(df_err.head())

    df_combined = df_concept.merge(df_err, on=['x'])
    df_combined = df_combined[['x', 'accuracy', 'mean_p_correct', 'c_target']]

    df_combined["err"] = 1-df_combined['']
    print(df_combined)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Create the scatter plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df_combined, x='mean_p_correct', y='accuracy', hue='c_target', style='c_target', s=100)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Moves legend outside the plot
    plt.xlabel('Probability of correct (1-err)')
    plt.ylabel('Accuracy in correct identification')
    plt.title('Accuracy vs Item simplicity')
    plt.show()
