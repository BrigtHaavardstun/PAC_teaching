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

def build_messages(C: list[int], x: int) -> Iterable[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                "You are a mathematical verifier. "
                "You will be given a list of questions of the form "
                "'Is x a multiple of k?'. "
                "Respond with a single line containing only a "
                "comma-separated list of T or F values, one per question, "
                "in order. Example: T,F,T,T,F. No other output."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [f"Is {x} a multiple of {c}?" for c in C]
            ),
        },
    ]


# ── Sampling ─────────────────────────────────────────────────────────────────

def observed_probs(
    tracker: UsageTracker,
    C: list[int],
    x: int,
    n_samples: int = 5,
) -> dict[str, float]:
    """
    Sequential sampling.

    Compatible with the updated UsageTracker where failed calls
    return None instead of raising exceptions.
    """

    messages = build_messages(C, x)
    ground_truth = [x % c == 0 for c in C]

    counts: Counter[str] = Counter()
    failures = 0

    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):

        text = tracker.chat(
            messages=messages,
            parse_fn=parse_with_mapping_wrapper(parse_fn=parse_err_extraction),
            query_type=QUERY_TYPE,
            metadata={
                "C": C,
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
    C: list[int],
    x: int,
    n_samples: int = 5,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, float]:
    """
    Concurrent sampling — fires all n_samples requests simultaneously.

    Requires tracker instantiated with AsyncOpenAI client.

    Semaphore caps total concurrency across entire experiment.
    """

    messages = build_messages(C, x)
    ground_truth = [x % c == 0 for c in C]

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
                    "C": C,
                    "x": x,
                    "ground_truth": ground_truth,
                },
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
    C: list[int],
    x_values: list[int],
    n_samples: int = 5,
    max_concurrent: int = 50,
):
    """
    Parallelises across all x values AND n_samples simultaneously.

    A single shared semaphore caps total in-flight requests
    across the entire experiment.
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_x(x: int):

        obs = await observed_probs_async(
            tracker,
            C,
            x,
            n_samples,
            semaphore,
        )

        ground_truth = [x % c == 0 for c in C]

        print(
            f"\n=== Results: {C=}, {x=}, n={n_samples} ==="
        )

        print(f"  Observed freq — {dict(obs)}")
        print(f"  Ground truth  — {ground_truth}")

    await asyncio.gather(
        *[_run_x(x) for x in x_values]
    )

    tracker.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_async():

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    tracker = UsageTracker(
        client,
        model="gpt-5-nano",
        log_path="usage_log.jsonl",
    )

    C = [13, 17, 19]

    x_values = list(
        range(0, 500)
    )

    await gen_output_async(
        tracker=tracker,
        C=C,
        x_values=x_values,
        n_samples=10,
        max_concurrent=15,
    )


def run():
    asyncio.run(run_async())


if __name__ == "__main__":

    run()
