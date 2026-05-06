import json
import asyncio
import datetime
import time
from pathlib import Path
from typing import Callable, Any, Iterable

from openai import (
    OpenAI,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

from openai.types.chat import ChatCompletionMessageParam
from openai._types import Omit
from openai.types.shared_params import ReasoningEffort


# ── Pricing table ($/million tokens) ────────────────────────────────────────
MODEL_PRICING = {
    "gpt-5-nano":   {"input": 0.05,  "output": 0.40},
    "gpt-5.4-nano": {"input": 0.20,  "output": 1.25},
    "gpt-5-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-5.4-nano-2026-03-17": {"input": 0.20,  "output": 1.25},
}


class UsageTracker:

    def __init__(
        self,
        client: OpenAI | AsyncOpenAI,
        model: str,
        log_path: str = "usage_log.jsonl",
        max_retries: int = 3,
        base_backoff: float = 1.0,
    ):
        self.client = client
        self.model = model
        self.log_path = Path(log_path)

        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            raise ValueError(f"No pricing entry for '{model}'.")

        self._price_in = pricing["input"]
        self._price_out = pricing["output"]

        self.max_retries = max_retries
        self.base_backoff = base_backoff

        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cost_usd = 0.0

    # ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        parse_fn: Callable[[str], Any] | None = None,
        query_type: str | None = None,
        metadata: dict | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 50,
        reasoning_effort: ReasoningEffort | Omit = "minimal",
    ) -> str | None:

        if not isinstance(self.client, OpenAI):
            raise RuntimeError(
                "chat() requires synchronous OpenAI client."
            )

        messages_list = self._safe_messages(messages)

        for attempt in range(self.max_retries + 1):

            try:

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_list,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                )

                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("Missing content.")

                text = content.strip()

                usage = response.usage
                if usage is None:
                    raise RuntimeError("Missing usage.")
                reasoning_tokens = "NA"
                if usage.completion_tokens_details:
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
                return self._log_success(
                    messages_list,
                    text,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    reasoning_tokens,
                    usage.total_tokens,
                    parse_fn,
                    query_type,
                    metadata,
                    reasoning_effort,
                )

            except BadRequestError as e:

                return self._log_error(
                    messages_list,
                    e,
                    query_type,
                    metadata,
                )

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:

                if attempt == self.max_retries:

                    return self._log_error(
                        messages_list,
                        e,
                        query_type,
                        metadata,
                    )

                backoff = self._compute_backoff(attempt)
                time.sleep(backoff)

        return None

    # ─────────────────────────────────────────────────────────────

    async def achat(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        parse_fn: Callable[[str], tuple[bool, Any]] | None = None,
        query_type: str | None = None,
        metadata: dict | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 600,
        reasoning_effort: ReasoningEffort | Omit = "none",
    ) -> str | None:

        if not isinstance(self.client, AsyncOpenAI):
            raise RuntimeError(
                "achat() requires AsyncOpenAI client."
            )

        messages_list = self._safe_messages(messages)

        for attempt in range(self.max_retries + 1):

            try:

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_list,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                )

                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("Missing content.")

                text = content.strip()

                usage = response.usage
                if usage is None:
                    raise RuntimeError("Missing usage.")

                reasoning_tokens = "NA"
                # Accessing via Python SDK
                if usage.completion_tokens_details:
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
                return self._log_success(
                    messages_list,
                    text,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    reasoning_tokens,
                    usage.total_tokens,
                    parse_fn,
                    query_type,
                    metadata,
                    reasoning_effort,
                )

            except BadRequestError as e:

                return self._log_error(
                    messages_list,
                    e,
                    query_type,
                    metadata,
                )

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:

                if attempt == self.max_retries:

                    return self._log_error(
                        messages_list,
                        e,
                        query_type,
                        metadata,
                    )

                backoff = self._compute_backoff(attempt)
                await asyncio.sleep(backoff)

        return None

    # ─────────────────────────────────────────────────────────────

    def _log_success(
        self,
        messages,
        text,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        total_tokens,
        parse_fn,
        query_type,
        metadata,
        reasoning_effort,
    ) -> str:

        cost = self._compute_cost(
            prompt_tokens,
            completion_tokens,
        )

        self.session_prompt_tokens += prompt_tokens
        self.session_completion_tokens += completion_tokens
        self.session_cost_usd += cost

        parsed = None
        parse_success = None

        if parse_fn is not None:

            try:
                sucess, parsed = parse_fn(text)
                parse_success = sucess

            except Exception:
                parsed = "N/A"
                parse_success = False

        entry = {
            "timestamp":
                datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),

            "model": self.model,
            "query_type": query_type,

            "messages": messages,
            "response": text,

            "parsed_response": parsed,
            "parse_success": parse_success,

            "metadata": metadata,

            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,

            "cost_usd": round(cost, 8),
            "reasoning_effort": reasoning_effort,
        }
        self._write_log(entry)

        return text

    # ─────────────────────────────────────────────────────────────

    def _log_error(
        self,
        messages,
        exception,
        query_type,
        metadata,
    ) -> None:

        error_entry = {
            "type": type(exception).__name__,
            "message": str(exception),
        }

        code = getattr(exception, "code", None)

        if code is not None:
            error_entry["code"] = code

        entry = {
            "timestamp":
                datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),

            "model": self.model,
            "query_type": query_type,

            "messages": messages,

            "response": None,
            "error": error_entry,

            "parsed_response": None,
            "parse_success": False,

            "metadata": metadata,

            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": 0.0,
        }

        self._write_log(entry)

        return None

    # ─────────────────────────────────────────────────────────────

    def _write_log(self, entry):

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ─────────────────────────────────────────────────────────────

    def _safe_messages(self, messages):

        try:

            return list(messages)

        except TypeError:

            return [m for m in messages]

    # ─────────────────────────────────────────────────────────────

    def _compute_backoff(self, attempt):

        return self.base_backoff * (2 ** attempt)

    # ─────────────────────────────────────────────────────────────

    def _compute_cost(
        self,
        prompt_tokens,
        completion_tokens,
    ):

        return (
            prompt_tokens / 1_000_000 * self._price_in
            +
            completion_tokens / 1_000_000 * self._price_out
        )

    # ─────────────────────────────────────────────────────────────

    def session_summary(self):

        return {
            "prompt_tokens":
                self.session_prompt_tokens,

            "completion_tokens":
                self.session_completion_tokens,

            "total_tokens":
                self.session_prompt_tokens
                +
                self.session_completion_tokens,

            "total_cost_usd":
                round(
                    self.session_cost_usd,
                    6,
                ),
        }

    def print_summary(self):

        s = self.session_summary()

        print("\n=== Session usage ===")

        print(
            f"  Prompt tokens:     {s['prompt_tokens']}"
        )

        print(
            f"  Completion tokens: {s['completion_tokens']}"
        )

        print(
            f"  Total tokens:      {s['total_tokens']}"
        )

        print(
            f"  Total cost:        "
            f"${s['total_cost_usd']:.6f}"
        )
