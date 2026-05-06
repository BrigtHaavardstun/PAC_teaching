# from huggingface_hub import login
from typing import List
# login(token=token)

import torch
import torch.nn.functional as F
import math
import pandas as pd
import gc
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def system_prompt(k):
    return f"You are a binary classifier. Determine whether the given number 'x' is a multiple of {k}, i.e. is x % k == 0. Reply with only 'Yes' or 'No', nothing else."


def run(model_id: str, pairs_k_x: List[tuple[int, int]]):
    # --- 1. SETUP & CONFIG ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # SLMs for 2026
    assert model_id in [
        "Qwen/Qwen3-4B-instruct-2507",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-4-e2b-it",
        "microsoft/Phi-4-mini-instruct",
    ]

    def no_think_prob(model, tokenizer, k, x):
        """Performs a single forward pass for intuitive probability."""
        messages = [
            {"role": "system", "content": system_prompt(k)},
            {"role": "user", "content": f"Is {x} a multiple of {k}?"},
            {"role": "assistant", "content": "Answer:"}
        ]

        prompt = tokenizer.apply_chat_template(messages, enable_thinking=False,
                                               tokenize=False, continue_final_message=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_logit = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(last_logit.float(), dim=-1)

        # Standard space-prefixed IDs for 2026 tokenizers
        yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
        no_ids = tokenizer.encode(" No",  add_special_tokens=False)

        assert len(yes_ids) == 1
        assert len(no_ids) == 1

        yes_id = yes_ids[0]
        no_id = no_ids[0]

        p_yes = math.exp(log_probs[yes_id].item())
        p_no = math.exp(log_probs[no_id].item())

        return p_yes, p_no

    # --- 2. MASTER EVALUATION LOOP ---
    master_data = []

    # Outer loop for models
    test_cases_pbar = tqdm(pairs_k_x, desc="Pairs Evaluated", position=0)

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    model.to(device)
    model.eval()

    for k, x in test_cases_pbar:
        test_cases_pbar.set_description(f"Current test case: {x=} {k=}")

        # Inner loop for test cases
        # we use leave=False so the nested bar disappears after each model

        py, pn = no_think_prob(model, tokenizer, k, x)

        master_data.append({
            "model": model_id,
            "k": k,
            "x": x,
            "p_yes": py,
            "p_no": pn,
            "sum": py + pn,
            "is_correct": (x % k == 0),
            "p_err": (1-py) if (x % k == 0) else (1-pn)
        })

    # --- 3. MEMORY PURGE ---
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- 4. EXPORT RESULTS ---
    df = pd.DataFrame(master_data)
    return df
