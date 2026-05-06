from typing import List
import torch
import torch.nn.functional as F
import math
import pandas as pd
import gc
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def system_prompt(concepts: List[int]) -> str:
    return (
        f"You are a version space learner. The concept class is multiples of k, "
        f"for k in {concepts}. Your task is to identify a correct k (possibly among many), "
        f"and report this. The user will give you two lists of numbers, first positive examples "
        f"(multiples of k), and then negative examples (not multiples of k). "
        f"Your answer should follow the format 'Answer: k=your-guess'"
    )


def _apply_chat_template(tokenizer, messages, enable_thinking: bool, continue_final_message: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            tokenize=False,
            continue_final_message=continue_final_message,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=continue_final_message,
        )


def _thinking_prefix_text(
    model,
    tokenizer,
    messages,
    device,
    think_end_token: str,
    max_think_new_tokens: int,
) -> str:
    think_messages = messages + [{"role": "assistant", "content": ""}]
    think_prompt_text = _apply_chat_template(
        tokenizer,
        think_messages,
        enable_thinking=True,
        continue_final_message=True,
    )

    think_inputs = tokenizer(think_prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        think_out = model.generate(
            **think_inputs,
            do_sample=False,
            max_new_tokens=max_think_new_tokens,
        )

    generated_ids = think_out[0, think_inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    end_idx = generated_text.find(think_end_token)
    if end_idx == -1:
        return generated_text
    return generated_text[:end_idx + len(think_end_token)]


def score_concepts(
    model, tokenizer, positive: List[int], negative: List[int],
    concepts: List[int], device, use_thinking: bool = False,
    think_end_token: str = "</think>", max_think_new_tokens: int = 2048,
    print_thinking: bool = False,
) -> dict[int, float]:
    """
    For each candidate k, computes P(str(k) + EOS | prompt) as a
    multi-token sequential product. Returns a dict mapping k -> probability.
    """
    positive_msg = str(positive)
    negative_msg = str(negative)
    if not positive:
        positive_msg = "No positive examples selected"
    else:
        positive_msg = "Positive: " + str(positive)
    if not negative:
        negative_msg = "No negative examples selected"
    else:
        negative_msg = "Negative: " + str(negative)
    base_messages = [
        {"role": "system", "content": system_prompt(concepts)},
        {"role": "user", "content": f"{str(positive_msg)}\n{str(negative_msg)}"},
    ]

    if use_thinking:
        thought_text = _thinking_prefix_text(
            model=model,
            tokenizer=tokenizer,
            messages=base_messages,
            device=device,
            think_end_token=think_end_token,
            max_think_new_tokens=max_think_new_tokens,
        )
        if print_thinking:
            print(f"[Thinking trace]\n{thought_text}\n")
        messages_for_scoring = base_messages + [{"role": "assistant", "content": f"{thought_text}\nAnswer: k="}]
        prompt_text = _apply_chat_template(
            tokenizer,
            messages_for_scoring,
            enable_thinking=False,
            continue_final_message=True,
        )
    else:
        messages_for_scoring = base_messages + [{"role": "assistant", "content": "Answer: k="}]
        prompt_text = _apply_chat_template(
            tokenizer,
            messages_for_scoring,
            enable_thinking=False,
            continue_final_message=True,
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    eos_id = tokenizer.eos_token_id

    probs = {}
    for k in concepts:
        option_ids = tokenizer.encode(str(k), add_special_tokens=False)

        # Full sequence: prompt + option tokens + EOS
        full_ids = prompt_ids + option_ids + [eos_id]
        input_tensor = torch.tensor([full_ids]).to(device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0].float()  # [seq_len, vocab]
            log_probs = F.log_softmax(logits, dim=-1)

        # logits[i] predicts token at position i+1, so
        # logits[prompt_len - 1 + i] predicts option_ids[i]
        total_log_prob = sum(
            log_probs[prompt_len - 1 + i, tok_id].item()
            for i, tok_id in enumerate(option_ids)
        )
        total_log_prob += log_probs[prompt_len - 1 + len(option_ids), eos_id].item()

        probs[k] = math.exp(total_log_prob)

    return probs


def run(
    model_id: str,
    test_cases_ps_ns_k: List[tuple[List[int], List[int], int]],
    concepts: List[int],
    use_thinking: bool = True,
    think_end_token: str = "</think>",
    max_think_new_tokens: int = 128,
    print_thinking: bool = True,
) -> pd.DataFrame:

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    assert model_id in [
        "Qwen/Qwen3-4B-instruct-2507",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-4-e2b-it",
        "microsoft/Phi-4-mini-instruct",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device)
    model.eval()

    master_data = []
    pbar = tqdm(test_cases_ps_ns_k, desc="Evaluating", position=0)

    for positive, negative, true_k in pbar:
        pbar.set_description(f"{positive=} {negative=} {true_k=}")

        concept_probs = score_concepts(
            model,
            tokenizer,
            positive,
            negative,
            concepts,
            device,
            use_thinking=use_thinking,
            think_end_token=think_end_token,
            max_think_new_tokens=max_think_new_tokens,
            print_thinking=print_thinking,
        )

        p_sum = sum(concept_probs.values())
        predicted_k = max(concept_probs, key=lambda x: concept_probs[x])

        master_data.append({
            "model":        model_id,
            "true_k":       true_k,
            "positive":     positive,
            "negative":     negative,
            "concepts": concepts,
            "concept_probs": [concept_probs[k] for k in concepts],  # ordered by concepts list
            "p_target": concept_probs[true_k],
            "p_non_answer":        1 - p_sum,
            "predicted_k":  predicted_k,
            "is_correct":   predicted_k == true_k,
            "p_not_correct":    1 - concept_probs[true_k],

        })

    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return pd.DataFrame(master_data)
