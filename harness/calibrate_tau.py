"""Gate1 tau calibration (pre-registered).

Generate rollouts for dataset indices [0, 32) with the INITIAL model
(training sampling config), compute ell_t = KL(p_T || p_ref) at every
non-masked completion position exactly as the A1 trainer path computes it,
pool all values, and set tau := median. Writes tau_calibration.json.
"""
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

import gate1_common as C

OUT = Path(__file__).resolve().parent.parent / "gate1_out" / "tau_calibration.json"
N_PROBLEMS = 32
SEED_BASE = 91000


def main():
    tok = C.load_tokenizer()
    model = C.load_base_model()
    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]

    all_ell = []
    t0 = time.time()
    for i in range(N_PROBLEMS):
        ex = ds[i]
        s_ids, t_ids, _ = C.build_prompts(tok, ex["problem"], ex["solution"])
        gen = C.generate_rollout(model, tok, s_ids, seed=SEED_BASE + i)
        seq = C.assemble(model, tok, s_ids, t_ids, gen)
        L = seq["sampled_token_ids"].shape[0]
        teacher_lp = C.forward_logprobs(model, seq["teacher_input_ids"], L, fp32_store=False)
        ref_logits_lp32 = C.forward_logprobs(model, seq["student_input_ids"], L, fp32_store=True)
        ref_lp = ref_logits_lp32.to(torch.bfloat16)
        ell = C.ell_fullkl(teacher_lp, ref_lp)
        mask = seq["shifted_labels"] != -100
        vals = ell.cpu()[mask.cpu()]
        all_ell.append(vals)
        print(f"[{i+1}/{N_PROBLEMS}] idx={i} tokens={int(mask.sum())} "
              f"ell_median={float(vals.median()):.3e} elapsed={time.time()-t0:.0f}s", flush=True)
        del teacher_lp, ref_lp, ref_logits_lp32, ell

    pooled = torch.cat(all_ell)
    q = torch.quantile(pooled, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99]))
    result = {
        "tau": float(q[1]),
        "rule": "median of ell_t over 32 initial-model rollouts (dataset idx 0..31)",
        "n_problems": N_PROBLEMS,
        "n_tokens": int(pooled.numel()),
        "ell_mean": float(pooled.mean()),
        "ell_p25": float(q[0]),
        "ell_p50": float(q[1]),
        "ell_p75": float(q[2]),
        "ell_p90": float(q[3]),
        "ell_p99": float(q[4]),
        "ell_max": float(pooled.max()),
        "frac_ell_gt_0.01": float((pooled > 0.01).float().mean()),
        "frac_ell_gt_0.1": float((pooled > 0.1).float().mean()),
        "frac_ell_eq_0": float((pooled == 0).float().mean()),
        "wall_seconds": time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)  # gate1_out is otherwise
    # only created by run_arm.sh, which runs after this calibration (README §4.1)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"TAU={result['tau']:.6e}  saved to {OUT}")


if __name__ == "__main__":
    main()
