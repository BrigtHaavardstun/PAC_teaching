import os
import asyncio
from collections import Counter
from tqdm.auto import tqdm
from openai import OpenAI, AsyncOpenAI

from openai.types.chat import ChatCompletionMessageParam
from OPENAI.tracker import UsageTracker
from OPENAI.parsers import parse_concept_identification
from typing import Iterable

QUERY_TYPE = "concept_identification"


# ── Prompt ───────────────────────────────────────────────────────────────────

def build_messages(C: list[int], positive: list[int], negative: list[int]) -> Iterable[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                f"You are given a set of candidate numbers: {C}. "
                "Exactly one of them is the target number. "
                "The user will provide two lists: positive examples, which are all exact multiples "
                "of the target, and negative examples, which are not multiples of the target. "
                "Your task is to identify which candidate is the target. "
                "Respond with only the number, nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"Positive examples: {positive}\nNegative examples: {negative}",
        },
    ]


# ── Sampling ─────────────────────────────────────────────────────────────────

def observed_probs(
    tracker: UsageTracker,
    C: list[int],
    c_target: int,
    positive: list[int],
    negative: list[int],
    n_samples: int = 5,
) -> dict[str, float]:
    """Sequential sampling. Use observed_probs_async for concurrent requests."""
    messages = build_messages(C, positive=positive, negative=negative)
    counts = Counter()

    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):
        text = tracker.chat(
            messages=messages,
            parse_fn=parse_concept_identification,
            query_type=QUERY_TYPE,
            reasoning_effort="minimal",
            metadata={
                "C":        C,
                "c_target": c_target,
                "positive": positive,
                "negative": negative,
            },
        )
        if text not in counts:
            print(text)
        counts[text] += 1

    total = sum(counts.values())
    return {resp: count / total for resp, count in counts.most_common()}


async def observed_probs_async(
    tracker: UsageTracker,
    C: list[int],
    c_target: int,
    positive: list[int],
    negative: list[int],
    n_samples: int = 5,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, float]:
    """
    Concurrent sampling — fires all n_samples requests simultaneously.
    Requires tracker to be instantiated with AsyncOpenAI client.
    Semaphore is shared across all experiments to cap total concurrency.
    """
    messages = build_messages(C, positive=positive, negative=negative)
    sem = semaphore if semaphore is not None else asyncio.Semaphore(n_samples)

    async def _call():
        async with sem:
            return await tracker.achat(
                messages=messages,
                parse_fn=parse_concept_identification,
                query_type=QUERY_TYPE,
                reasoning_effort="minimal",
                metadata={
                    "C":        C,
                    "c_target": c_target,
                    "positive": positive,
                    "negative": negative,
                },
            )

    results = await asyncio.gather(*[_call() for _ in range(n_samples)])
    counts: Counter = Counter(results)

    total = sum(counts.values())
    return {resp: count / total for resp, count in counts.most_common()}


def gen_output(
    tracker: UsageTracker,
    C: list[int],
    c_target: int,
    positive: list[int],
    negative: list[int],
    n_samples: int = 5,
):
    obs = observed_probs(
        tracker=tracker, C=C, c_target=c_target,
        positive=positive, negative=negative, n_samples=n_samples,
    )
    print(f"\n=== Results: {C=}, {positive=}, {negative=}, n={n_samples} ===")
    print(f"  Observed freq — {dict(obs)}")
    print(f"  Ground truth  — {c_target}")
    tracker.print_summary()


async def gen_output_async(
    tracker: UsageTracker,
    C: list[int],
    experiments: list[dict],
    n_samples: int = 5,
    max_concurrent: int = 50,
):
    """
    Parallelises across all experiments AND n_samples simultaneously.
    Each experiment is a dict with keys: c_target, positive, negative.

    Example:
        experiments = [
            {"c_target": 17, "positive": [697], "negative": []},
            {"c_target": 13, "positive": [650], "negative": []},
        ]
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_experiment(exp: dict):
        c_target = exp["c_target"]
        positive = exp["positive"]
        negative = exp["negative"]
        obs = await observed_probs_async(
            tracker=tracker, C=C, c_target=c_target,
            positive=positive, negative=negative,
            n_samples=n_samples, semaphore=semaphore,
        )
        print(f"\n=== Results: {C=}, {positive=}, {negative=}, n={n_samples} ===")
        print(f"  Observed freq — {dict(obs)}")
        print(f"  Ground truth  — {c_target}")

    await asyncio.gather(*[_run_experiment(exp) for exp in experiments])
    tracker.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_async():
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tracker = UsageTracker(client, model="gpt-5-nano", log_path="usage_log.jsonl")

    C = [13, 17, 19]
    X = list(range(0, 400))
    unique_id_positive = [x for x in X if sum(x % c == 0 for c in C) == 1]
    experiments = []
    for x in unique_id_positive:
        c_target = None
        for c in C:
            if x % c == 0:
                assert c_target is None, f"{c_target=} and {c=} for {x=}"
                c_target = c
        experiments.append(
            {"c_target": c_target, "positive": [x], "negative": []})
    await gen_output_async(tracker=tracker, C=C, experiments=experiments, n_samples=10, max_concurrent=20)


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
