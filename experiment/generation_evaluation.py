import torch
import torch.nn.functional as F
import math
from collections import Counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def system_prompt(k):
    return (
        f"You are a binary classifier. Determine whether the given number 'x' "
        f"is a multiple of {k}, i.e. is x % k == 0. Reply with only 'Yes' or 'No', nothing else."
    )


def build_inputs(tokenizer, k, x, device):
    messages = [
        {"role": "system",    "content": system_prompt(k)},
        {"role": "user",      "content": f"Is {x} a multiple of {k}?"},
        {"role": "assistant", "content": "Answer:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,
        tokenize=False,
        continue_final_message=True,
    )
    return tokenizer(prompt, return_tensors="pt").to(device)


def predicted_probs(model, tokenizer, config: GenerationConfig, inputs):
    """Single forward pass → p(Yes) and p(No) over the full vocabulary."""
    with torch.no_grad():
        logits = model(**inputs, config=config).logits[0, -1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)

    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No",  add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, \
        "' Yes' or ' No' tokenises to more than one token — adjust your target strings."

    p_yes = math.exp(log_probs[yes_ids[0]].item())
    p_no = math.exp(log_probs[no_ids[0]].item())
    return p_yes, p_no


def observed_probs(model, tokenizer, config: GenerationConfig, inputs, n_samples=500):
    """
    Run generate() n_samples times with pure sampling (temp=1, no truncation).
    Count how each output is produced.
    """
    prompt_len = inputs["input_ids"].shape[1]
    counts = Counter()

    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=config)

        response = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        counts[response] += 1

    total = sum(counts.values())
    return {response: count / total for response, count in counts.most_common()}


def sanity_check(model, tokenizer, device, k, x, n_samples=50):

    pure_sampling = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,           # no top-k filtering
        top_p=1.0,         # full distribution
        max_new_tokens=1,  # we only care about the single next token
    )
    inputs = build_inputs(tokenizer, k, x, device)

    # --- Predicted (forward pass) ---
    p_yes, p_no = predicted_probs(model, tokenizer, config=pure_sampling, inputs=inputs)
    p_yes_renorm = p_yes / (p_yes + p_no)   # renormalised over just Yes/No
    p_no_renorm = p_no / (p_yes + p_no)

    # --- Observed (sampling) ---
    obs = observed_probs(model, tokenizer, config=pure_sampling, inputs=inputs, n_samples=n_samples)

    print(f"\n=== Sanity check: k={k}, x={x}, n={n_samples} ===")
    print(f"  Forward pass  — p(Yes)={p_yes:.4f}, p(No)={p_no:.4f}  "
          f"[renorm: Yes={p_yes_renorm:.3f}, No={p_no_renorm:.3f}]")
    print(f"  Observed freq — {dict(obs)}")
    print(f"  Ground truth  — {x} % {k} == 0 → {x % k == 0}")


def run():
    model_id = "Qwen/Qwen3-4B-instruct-2507"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=None,
    ).to(device).eval()
    for k in [2394727983]:
        for x in [5483924572866]:
            sanity_check(model=model, tokenizer=tokenizer, device=device, k=k, x=x, n_samples=100)

#  Forward pass  — p(Yes)=0.1823, p(No)=0.8171  [renorm: Yes=0.182, No=0.818]
# Observed freq — {'No': 0.831, 'Yes': 0.168, '': 0.001}
# Ground truth  — 343 % 17 == 0 → False


# === Sanity check: k=2394727983, x=5483924572866, n=100 ===
# Forward pass  — p(Yes)=0.3775, p(No)=0.6224  [renorm: Yes=0.378, No=0.622]
#  Observed freq — {'No': 0.66, 'Yes': 0.34}
#  Ground truth  — 5483924572866 % 2394727983 == 0 → False
if __name__ == "__main__":
    #  Forward pass  — p(Yes)=0.1823, p(No)=0.8171  [renorm: Yes=0.182, No=0.818]
    # Observed freq — {'No': 0.831, 'Yes': 0.168, '': 0.001}
    # Ground truth  — 343 % 17 == 0 → False
    run()
