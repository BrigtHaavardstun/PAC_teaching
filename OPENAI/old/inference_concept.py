import os
import json
import datetime
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm
from openai import OpenAI
from typing import List


# ── Pricing table ($/million tokens) ────────────────────────────────────────
MODEL_PRICING = {
    "gpt-5-nano":   {"input": 0.05,  "output": 0.40},
    "gpt-5.4-nano": {"input": 0.20,  "output": 1.25},
    "gpt-5-mini":   {"input": 0.40,  "output": 1.60},
}


class UsageTracker:
    """
    Wraps an OpenAI client to track token usage and cost per call,
    and persists every Q&A exchange to a JSONL log file.
    """

    def __init__(self, client: OpenAI, model: str, log_path: str = "usage_log.jsonl"):
        self.client = client
        self.model = model
        self.log_path = Path(log_path)

        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            raise ValueError(f"No pricing entry for '{model}'. Add it to MODEL_PRICING.")
        self._price_in = pricing["input"]   # $ per 1M tokens
        self._price_out = pricing["output"]

        # Session-level accumulators
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cost_usd = 0.0

    # ── Core call ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        metadata: dict = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 600,
        reasoning_effort="minimal"
    ) -> str:
        """
        Drop-in replacement for client.chat.completions.create().
        Returns the response text; logs everything to disk.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort
        )

        text = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = self._compute_cost(prompt_tokens, completion_tokens)

        # Update session totals
        self.session_prompt_tokens += prompt_tokens
        self.session_completion_tokens += completion_tokens
        self.session_cost_usd += cost

        # Persist to JSONL
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),            "model":             self.model,
            "messages":          messages,          # full input
            "response":          text,              # full output
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "cost_usd":          round(cost, 8),
            "reasoning_effort": reasoning_effort
        }
        if metadata:
            entry["metadata"] = metadata

        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        return text

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000 * self._price_in +
            completion_tokens / 1_000_000 * self._price_out
        )

    def session_summary(self) -> dict:
        return {
            "prompt_tokens":     self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens":      self.session_prompt_tokens + self.session_completion_tokens,
            "total_cost_usd":    round(self.session_cost_usd, 6),
        }

    def print_summary(self):
        s = self.session_summary()
        print("\n=== Session usage ===")
        print(f"  Prompt tokens:     {s['prompt_tokens']}")
        print(f"  Completion tokens: {s['completion_tokens']}")
        print(f"  Total tokens:      {s['total_tokens']}")
        print(f"  Total cost:        ${s['total_cost_usd']:.6f}")


# ── Prompt helpers ───────────────────────────────────────────────────────────

def system_prompt(C):
    return (
        f"You are given a set of candidate numbers: {C}. "
        "Exactly one of them is the target number. "
        "The user will provide two lists: positive examples, which are all exact multiples of the target, "
        "and negative examples, which are not multiples of the target. "
        "Your task is to identify which candidate is the target. "
        "Respond with only the number, nothing else."
    )


def build_messages(C: list[int], positive: list[int], negative: list[int]) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(C)},
        {"role": "user",   "content": f"Positive examples: {positive}\nNegative exampels: {negative}"},
    ]


# ── Sampling ─────────────────────────────────────────────────────────────────

def observed_probs(tracker: UsageTracker, c_target: int, C: list[int], positive: list[int], negative: list[int],  n_samples: int = 5):
    messages = build_messages(C, positive=positive, negative=negative)
    counts = Counter()

    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):
        text = tracker.chat(
            messages=messages,
            metadata={"C": C, "c*": c_target, "positive": positive, "negative": negative},
            temperature=1.0,
            top_p=1.0,
            max_completion_tokens=200,
        )
        if text not in counts:
            print(text)
        counts[text] += 1

    total = sum(counts.values())
    return {resp: count / total for resp, count in counts.most_common()}


def gen_output(tracker: UsageTracker, C: list[int], c_target: int, positive: list[int], negative: list[int], n_samples: int = 5):
    obs = observed_probs(tracker=tracker, C=C, c_target=c_target,
                         positive=positive, negative=negative, n_samples=n_samples)
    ground_truth = c_target

    print(f"\n=== Results: {C=}, {positive=}, {negative=}, n={n_samples} ===")
    print(f"  Observed freq — {dict(obs)}")
    print(f"  Ground truth  — {ground_truth}")
    tracker.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tracker = UsageTracker(client, model="gpt-5-nano", log_path="usage_log.jsonl")

    C = [7, 11, 13, 17, 19, 23]
    # x = 163739394 Fails
    # x = 39734415077 Works well!
    # x = 49 God example!
    # x = 203 God example!
    positive = [203]
    negative = []
    gen_output(tracker=tracker, C=C, c_target=7, positive=positive, negative=negative, n_samples=50)


if __name__ == "__main__":
    run()
