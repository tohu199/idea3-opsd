"""Shared helpers for gate1 standalone scripts (calibration / eval / diagnostic).

Prompt construction and sequence assembly mirror the OPSD training path
exactly (SelfDistillationDataCollator + OPSDTrainer.training_step), so the
signals measured here are the same quantities the trainer computes.
"""
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "Qwen/Qwen3-1.7B"
TEMPERATURE = 1.1  # training temperature (official 1.7B script)
MAX_NEW_TOKENS = 512


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.cuda()
    model.eval()
    return model


def sampling_generation_config(tok):
    return GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        do_sample=True,
        top_k=20,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )


def build_prompts(tok, problem, solution):
    """Return (student_prompt_text, teacher_prompt_text) exactly as the collator does."""
    from data_collator import SelfDistillationDataCollator

    collator = SelfDistillationDataCollator(
        tokenizer=tok, max_length=2048, reason_first=False,
        student_thinking=False, teacher_thinking=True,
    )
    feats = collator([{"problem": problem, "solution": solution}])
    student_ids = feats["student_prompts"][0]
    teacher_ids = feats["teacher_prompts"][0]
    return student_ids, teacher_ids, feats


@torch.no_grad()
def generate_rollout(model, tok, student_prompt_ids, seed):
    """Generate one on-policy rollout (batch 1) with the training sampling config.

    transformers 4.57 routes a per-call `generator` kwarg into model_kwargs
    validation and rejects it, so the seed is set on the global RNG
    immediately before generation instead (equivalent for a single call).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = student_prompt_ids.unsqueeze(0).cuda()
    attn = torch.ones_like(ids)
    out = model.generate(
        input_ids=ids,
        attention_mask=attn,
        generation_config=sampling_generation_config(tok),
        return_dict_in_generate=True,
        use_cache=True,
    )
    return out.sequences[0].cpu()  # (prompt_len + gen_len,)


def assemble(model, tok, student_prompt_ids, teacher_prompt_ids, generated_ids):
    """Mirror training_step: build full sequences + shifted labels.

    Returns dict with student_input_ids, teacher_input_ids, labels(shifted),
    sampled_token_ids; all 1-D tensors on CPU.
    """
    student_prompt_len = student_prompt_ids.shape[0]
    teacher_prompt_len = teacher_prompt_ids.shape[0]
    generation_ids = generated_ids[student_prompt_len:]
    student_full = generated_ids.clone()
    teacher_full = torch.cat([teacher_prompt_ids.cpu(), generation_ids])
    labels = generated_ids.clone()
    labels[:student_prompt_len] = -100
    if tok.pad_token_id is not None:
        labels[labels == tok.pad_token_id] = -100
    shifted_labels = labels[student_prompt_len:]
    return {
        "student_input_ids": student_full,
        "teacher_input_ids": teacher_full,
        "teacher_prompt_len": teacher_prompt_len,
        "student_prompt_len": student_prompt_len,
        "shifted_labels": shifted_labels,
        "sampled_token_ids": generation_ids,
    }


@torch.no_grad()
def forward_logprobs(model, input_ids, n_positions, temperature=TEMPERATURE, fp32_store=False):
    """Log-probs at the last n_positions predicted positions (temperature-scaled).

    Uses logits_to_keep = n_positions + 1 and returns log-probs for the first
    n_positions of those, i.e. the distributions predicting
    input_ids[-n_positions:]. Mirrors the (patched) trainer forwards.
    """
    ids = input_ids.unsqueeze(0).cuda()
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, logits_to_keep=n_positions + 1)
    logits = out.logits[0, :n_positions, :]
    if fp32_store:
        lp = F.log_softmax(logits.float() / temperature, dim=-1)
    else:
        lp = F.log_softmax(logits / temperature, dim=-1)
    return lp  # (n_positions, vocab) on cuda


def ell_fullkl(teacher_log_probs_bf16, ref_log_probs_bf16, chunk=64):
    """ell_t = KL(p_T || p_ref) per position, fp32 chunked (mirrors trainer A1)."""
    L = teacher_log_probs_bf16.shape[0]
    ell = torch.empty(L, dtype=torch.float32, device=teacher_log_probs_bf16.device)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        tp = teacher_log_probs_bf16[s:e].float()
        rp = ref_log_probs_bf16[s:e].float()
        ell[s:e] = (tp.exp() * (tp - rp)).sum(dim=-1)
    return ell


def extract_boxed(text):
    """Return inner content of the LAST \\boxed{...} (balanced braces), or None."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def normalize_answer(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(",", "").replace(" ", "").replace("\n", "")
    if s.endswith("."):
        s = s[:-1]
    return s.lower()


def _as_number(s):
    """Parse a normalized answer as a float (handles simple p/q fractions)."""
    if s is None:
        return None
    s = s.replace("\\frac", "FRAC")  # LaTeX fractions won't parse; handled by string match
    if s.startswith("FRAC"):
        return None
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        den = int(m.group(2))
        return None if den == 0 else int(m.group(1)) / den
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(pred_raw, gold_raw):
    p, g = normalize_answer(pred_raw), normalize_answer(gold_raw)
    if p is None or g is None:
        return False
    if p == g:
        return True
    pn, gn = _as_number(p), _as_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= 1e-6 * max(1.0, abs(gn))
    return False
