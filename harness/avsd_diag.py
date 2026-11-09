"""Gate1 AVSD blind-spot diagnostic (pre-registered definition).

Held-out indices [960, 1060). For each problem, take ONE student rollout
from the initial model (training sampling config), then score it under
three teacher views built from the single reference solution:
  full    = reference solution as-is
  partial = first half of the reference solution (by characters)
  answer  = "The final answer is \\boxed{<gold>}." only
plus the privilege-free reference (student context) for ell_t.

Per completion position:
  ell_t = KL(p_T^full || p_ref)            (same computation as A1)
  J_t   = KL(q^A_t || q~^G_t), q^A = arithmetic mean of the 3 view
          distributions, q~^G = normalized geometric mean.
All distributions are the temperature-1.1 scaled ones used in training;
view log-probs are computed in fp32 from the bf16 logits for the pooling.

Pre-registered read: among positions with ell in the top 25%, the fraction
with J at or below the pooled median of J. >= 50% means AVSD's gate misses
the majority of high-leakage positions.

Outputs avsd_diag.json (summary + per-problem stats) and avsd_diag.npz
(pooled per-position ell/J arrays).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

import gate1_common as C

OUT_DIR = Path(__file__).resolve().parent.parent / "gate1_out"
START, END = 960, 1060
SEED_BASE = 92000


def view_solutions(solution):
    gold = C.extract_boxed(solution)
    partial = solution[: len(solution) // 2]
    if gold is not None:
        answer = f"The final answer is \\boxed{{{gold}}}."
    else:
        answer = solution[-200:]
    return {"full": solution, "partial": partial, "answer": answer}, gold


@torch.no_grad()
def view_logprobs(model, tok, problem, view_solution, generation_ids, student_prompt_len):
    """fp32 log-probs of a teacher view at completion positions."""
    _, t_ids, _ = C.build_prompts(tok, problem, view_solution)
    full = torch.cat([t_ids, generation_ids])
    L = generation_ids.shape[0]
    ids = full.unsqueeze(0).cuda()
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, logits_to_keep=L + 1)
    logits = out.logits[0, :L, :]
    return F.log_softmax(logits.float() / C.TEMPERATURE, dim=-1)


def compute_J(lp_list, chunk=32):
    """J_t = KL(q^A || q~^G) per position from a list of fp32 log-prob tensors."""
    L = lp_list[0].shape[0]
    J = torch.empty(L, dtype=torch.float32, device=lp_list[0].device)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        lps = torch.stack([lp[s:e] for lp in lp_list])  # (M, n, V)
        log_qA = torch.logsumexp(lps, dim=0) - np.log(len(lp_list))
        log_qG_raw = lps.mean(dim=0)
        log_qG = log_qG_raw - torch.logsumexp(log_qG_raw, dim=-1, keepdim=True)
        qA = log_qA.exp()
        J[s:e] = (qA * (log_qA - log_qG)).sum(dim=-1)
    return J


def main():
    tok = C.load_tokenizer()
    model = C.load_base_model()
    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]

    all_ell, all_J = [], []
    per_problem = []
    t0 = time.time()
    for k, idx in enumerate(range(START, END)):
        ex = ds[idx]
        views, gold = view_solutions(ex["solution"])
        s_ids, t_ids, _ = C.build_prompts(tok, ex["problem"], ex["solution"])
        gen = C.generate_rollout(model, tok, s_ids, seed=SEED_BASE + idx)
        seq = C.assemble(model, tok, s_ids, t_ids, gen)
        gen_ids = seq["sampled_token_ids"]
        L = gen_ids.shape[0]
        mask = (seq["shifted_labels"] != -100).cuda()

        # ell_t: full-view teacher (bf16 path, as in training) vs ref (bf16-stored)
        teacher_lp_bf16 = C.forward_logprobs(model, seq["teacher_input_ids"], L, fp32_store=False)
        ref_lp = C.forward_logprobs(model, seq["student_input_ids"], L, fp32_store=True).to(torch.bfloat16)
        ell = C.ell_fullkl(teacher_lp_bf16, ref_lp)

        # J_t from the three views (fp32 log-probs)
        lps = [
            view_logprobs(model, tok, ex["problem"], views[v], gen_ids, seq["student_prompt_len"])
            for v in ("full", "partial", "answer")
        ]
        J = compute_J(lps)

        ell_m = ell[mask].cpu()
        J_m = J[mask].cpu()
        all_ell.append(ell_m)
        all_J.append(J_m)
        per_problem.append({
            "index": idx, "n_tokens": int(mask.sum()),
            "ell_mean": float(ell_m.mean()), "J_mean": float(J_m.mean()),
        })
        print(f"[{k+1}/{END-START}] idx={idx} tokens={int(mask.sum())} "
              f"ell_mean={float(ell_m.mean()):.4f} J_mean={float(J_m.mean()):.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        del teacher_lp_bf16, ref_lp, ell, lps, J

    ell_all = torch.cat(all_ell)
    J_all = torch.cat(all_J)
    ell_thr = float(torch.quantile(ell_all, 0.75))
    J_med = float(J_all.median())
    high = ell_all >= ell_thr
    frac = float((J_all[high] <= J_med).float().mean())
    corr = float(torch.corrcoef(torch.stack([ell_all, J_all]))[0, 1])
    summary = {
        "indices": [START, END],
        "n_problems": END - START,
        "n_positions": int(ell_all.numel()),
        "ell_p75_threshold": ell_thr,
        "J_median": J_med,
        "frac_high_ell_low_J": frac,
        "preregistered_read": ">= 0.50 means AVSD's gate misses the majority of high-leakage positions",
        "pearson_corr_ell_J": corr,
        "ell_mean": float(ell_all.mean()),
        "J_mean": float(J_all.mean()),
        "wall_seconds": time.time() - t0,
        "per_problem": per_problem,
    }
    (OUT_DIR / "avsd_diag.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT_DIR / "avsd_diag.npz",
             ell=ell_all.numpy(), J=J_all.numpy())
    print(json.dumps({k: v for k, v in summary.items() if k != "per_problem"}, indent=2))
    print(f"FRAC_HIGH_ELL_LOW_J={frac:.4f}  saved to {OUT_DIR}/avsd_diag.json")


if __name__ == "__main__":
    main()
