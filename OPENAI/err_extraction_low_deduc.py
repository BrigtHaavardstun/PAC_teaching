import os
import asyncio
from collections import Counter
from tqdm.auto import tqdm
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletionMessageParam
from OPENAI.tracker import UsageTracker
from OPENAI.parsers import parse_err_extraction, parse_with_mapping_wrapper
from typing import Iterable

QUERY_TYPE = "err_extraction"


# ── Prompt ───────────────────────────────────────────────────────────────────

def build_messages(c: int, x: int) -> Iterable[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": "Answer format: 'Answer = Yes' or 'Answer = No'",
        },
        {
            "role": "user",
            "content": f"Is {c} a divisor of {x}?"
        },
    ]


# ── Sampling ─────────────────────────────────────────────────────────────────

def observed_probs(
    tracker: UsageTracker,
    c: int,
    x: int,
    n_samples: int = 5,
) -> dict[str, float]:
    """
    Sequential sampling.

    Compatible with the updated UsageTracker where failed calls
    return None instead of raising exceptions.
    """

    messages = build_messages(c, x)
    ground_truth = x % c == 0

    counts: Counter[str] = Counter()
    failures = 0

    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):

        text = tracker.chat(
            messages=messages,
            parse_fn=parse_err_extraction,
            query_type=QUERY_TYPE,
            metadata={
                "c": c,
                "x": x,
                "ground_truth": ground_truth,
            },
        )

        if text is None:
            failures += 1
            continue

        if text not in counts:
            print(text)

        counts[text] += 1

    total = sum(counts.values())

    if failures > 0:
        print(f"Failures: {failures}/{n_samples}")

    if total == 0:
        return {}

    return {
        resp: count / total
        for resp, count in counts.most_common()
    }


async def observed_probs_async(
    tracker: UsageTracker,
    c: int,
    x: int,
    n_samples: int = 5,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, float]:
    """
    Concurrent sampling — fires all n_samples requests simultaneously.

    Requires tracker instantiated with AsyncOpenAI client.

    Semaphore caps total concurrency across entire experiment.
    """

    messages = build_messages(c, x)
    ground_truth = x % c == 0

    sem = (
        semaphore
        if semaphore is not None
        else asyncio.Semaphore(n_samples)
    )

    async def _call():

        async with sem:

            return await tracker.achat(
                messages=messages,
                parse_fn=parse_err_extraction,
                query_type=QUERY_TYPE,
                metadata={
                    "c": c,
                    "x": x,
                    "ground_truth": ground_truth,
                },
                reasoning_effort="none"
            )

    results = await asyncio.gather(
        *[_call() for _ in range(n_samples)]
    )

    valid_results = [
        r for r in results
        if r is not None
    ]

    failures = n_samples - len(valid_results)

    counts: Counter[str] = Counter(valid_results)

    total = sum(counts.values())

    if failures > 0:
        print(f"Failures: {failures}/{n_samples}")

    if total == 0:
        return {}

    return {
        resp: count / total
        for resp, count in counts.most_common()
    }


async def gen_output_async(
    tracker: UsageTracker,
    experiments: list[dict],
    n_samples: int = 5,
    max_concurrent: int = 50,
):
    """
    Parallelises across all x values AND n_samples simultaneously.

    A single shared semaphore caps total in-flight requests
    across the entire experiment.
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_x(c: int, x: int):

        obs = await observed_probs_async(
            tracker,
            c,
            x,
            n_samples,
            semaphore,
        )

        ground_truth = x % c == 0

        print(
            f"\n=== Results: {c=}, {x=}, n={n_samples} ==="
        )

        print(f"  Observed freq — {dict(obs)}")
        print(f"  Ground truth  — {ground_truth}")

    await asyncio.gather(
        *[_run_x(c=exp["c"], x=exp["x"]) for exp in experiments]
    )

    tracker.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_async():

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    tracker = UsageTracker(
        client,
        model="gpt-5.4-nano-2026-03-17",
        log_path="usage_log_td2.jsonl"
    )

    import json

    C = [5, 7, 11, 13, 17]
    import pandas as pd
    df = pd.read_csv("Exerpiment_TS.csv")
    X = set(df["positive"].unique()) | set(df["negative"].unique())
    X = sorted(list(map(int, X)))

    experiments = []
    for x in X:
        for c in C:
            experiments.append(
                {"c": c, "x": x})

    await gen_output_async(
        tracker=tracker,
        experiments=experiments,
        n_samples=10,
        max_concurrent=30,
    )


def run():
    asyncio.run(run_async())


if __name__ == "__main__":

    run()
