# PAC Teaching Experiments

This repository contains the research code, experiment logs, and generated figures associated with a paper on teaching set construction for language-model learners under deductive error.

The project studies a simple but useful concept class: divisibility / multiples-of-`k` concepts. The experiments ask how errors in low-level deductive judgements, such as whether `c` divides `x`, affect downstream concept identification and teaching performance.

> Repository status: this is a research artifact, not a polished software package. It currently preserves the code and outputs close to the state used for the paper. Some scripts are exploratory, contain hard-coded experiment settings, or were edited during different experiment runs.

## What Is Included

- `OPENAI/` — main OpenAI API experiment and analysis code.
- `select_ts/` — teaching-set selection utilities and heuristics.
- `experiment/` — local model / version-space learner experiments using Hugging Face models.
- `LLM/` — notebooks, CSV outputs, and plots from local or multi-model experiments.
- `Results/` — derived result tables, figures, and comparison outputs.
- `*.jsonl` — raw OpenAI usage logs with prompts, responses, parsed values, metadata, token counts, and costs.
- `*.png` — generated figures used during analysis and paper preparation.
- `run_experiments.py` — simple command dispatcher for the main OpenAI experiment and analysis scripts.

## Research Workflow

At a high level, the workflow is:

1. Select candidate teaching examples for divisibility concepts.
2. Query language models on deductive tasks, e.g. “Is `c` a divisor of `x`?”
3. Query language models on concept-identification tasks from positive and negative examples.
4. Parse and log model responses.
5. Analyze how deductive error predicts or explains teaching / concept-identification error.
6. Generate plots and tables for the paper.

The most relevant paper-facing scripts are currently in `OPENAI/` and `select_ts/`.

## Main Commands

The top-level dispatcher supports the following commands:

```bash
python run_experiments.py err
python run_experiments.py concept
python run_experiments.py err_analysis
python run_experiments.py concept_analysis
python run_experiments.py combined_analysis
python run_experiments.py new_analysis
python run_experiments.py 2td
```

Important caveat: several scripts currently use hard-coded model names, log-file names, sample counts, and input CSVs. Before re-running experiments, inspect the relevant script and confirm the settings match the experiment you intend to reproduce.

## Environment

The code is Python-based. The main dependencies used across the repository include:

- `openai`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `tqdm`
- `sympy`
- `torch`
- `transformers`
- `huggingface_hub`

For OpenAI API experiments, set:

```bash
export OPENAI_API_KEY="..."
```

Local virtual environments are intentionally ignored by Git. Recreate your own environment before running experiments.

## Data and Logs

The `.jsonl` files are experiment logs. Each line is a JSON record containing information such as:

- model name
- query type
- prompt messages
- raw response
- parsed response
- parse status
- experiment metadata
- token usage and estimated cost

These logs are useful for reproducing analyses without re-querying models.

## Reproducibility Notes

This repository was initialized from the working research directory after the paper experiments had been run. The tag `baseline-as-is` points to the initial snapshot before cleanup.

To return to that snapshot:

```bash
git checkout baseline-as-is
```

Because this is a research codebase, exact reproduction may require checking the hard-coded settings in individual scripts against the experiment described in the paper.

## Citation

If you use this repository, please cite the associated research paper. A full citation entry will be added once the paper metadata is finalized.
