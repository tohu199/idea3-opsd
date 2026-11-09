"""Fair-A2 tau calibration (pre-registered rule; see docs/fair_a2_calibration.md).

Same 32 initial-model rollouts as calibrate_tau.py (dataset idx 0..31,
SEED_BASE 91000), but the signal is A2's sampled-token Delta instead of
A1's full-vocab KL ell_t:
    Delta_t = log p_T(y_t) - log p_ref(y_t)   (temperature-scaled, at the
    sampled token y_t; exactly the trainer's "sampled" mode).

Why a new rule is needed: ell_t >= 0, so its median is a usable scale.
Delta_t is signed, so its plain median can be <= 0 and is unusable as a
tau (a non-positive tau inverts the weighting). Two scale candidates are
reported:
  * tau_median_abs: median(|Delta_t|) — the direct analogue of the ell
    median rule for a signed signal.
  * tau_ess: the tau > 0 whose pooled weights w = exp(-Delta/tau) have
    ESS ratio equal to A1's Gate-1 mean per-micro-batch ESS ratio
    (ESS_TARGET_A1 = 0.479, docs/gate1_results.md). This is the
    RECOMMENDED tau: "fair" here means the same effective learning mass
    as A1, so only the signal — not the amount of down-weighting —
    differs between A1 and fair A2.

Writes gate1_out/tau_calibration_delta.json. GPU run, ~8 min like §4.1.
"""
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset

import gate1_common as C

OUT = Path(__file__).resolve().parent.parent / "gate1_out" / "tau_calibration_delta.json"
N_PROBLEMS = 32
SEED_BASE = 91000
# A1 Gate-1 mean per-micro-batch ESS ratio (docs/gate1_results.md: 0.4793).
ESS_TARGET_A1 = 0.479
# Gate-1 tau (ell median), kept only to document the collapse at that value.
OLD_TAU = 5.306877e-04


# --- pure-python math on lists of floats (no GPU; unit-testable) ---

def _quantile(sorted_xs, q):
    n = len(sorted_xs)
    if n == 0:
        return float("nan")
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def _median(xs):
    return _quantile(sorted(xs), 0.5)


def ess_ratio(deltas, tau):
    """ESS ratio of w = exp(-Delta/tau) over the pooled tokens.

    (sum w)^2 / (n * sum w^2). Invariant to a common scale factor, so the
    exponents are max-shifted (same stabilisation as patch_opsd3.py).
    """
    n = len(deltas)
    if n == 0:
        return 0.0
    exps = [-d / tau for d in deltas]
    m = max(exps)
    ws = [math.exp(e - m) for e in exps]
    s1 = sum(ws)
    s2 = sum(w * w for w in ws)
    return (s1 * s1) / (n * s2)


def solve_tau_for_ess(deltas, target, lo=1e-6, hi=1e4, iters=100):
    """Smallest tau with ess_ratio(deltas, tau) >= target (bisection on log tau).

    ESS ratio increases monotonically to 1 as tau grows. Returns None when
    the signal is degenerate (all deltas equal -> ESS is 1 for every tau).
    """
    if not deltas or max(deltas) - min(deltas) < 1e-12:
        return None
    if ess_ratio(deltas, hi) <= target:
        return hi
    if ess_ratio(deltas, lo) >= target:
        return lo
    lo_l, hi_l = math.log10(lo), math.log10(hi)
    for _ in range(iters):
        mid_l = (lo_l + hi_l) / 2.0
        if ess_ratio(deltas, 10.0 ** mid_l) < target:
            lo_l = mid_l
        else:
            hi_l = mid_l
    return 10.0 ** ((lo_l + hi_l) / 2.0)


def summarize(deltas):
    """All calibration candidates + diagnostics from the pooled Delta list."""
    s = sorted(deltas)
    n = len(deltas)
    abs_deltas = [abs(d) for d in deltas]
    tau_med = _median(deltas)
    tau_med_abs = _median(abs_deltas)
    tau_ess = solve_tau_for_ess(deltas, ESS_TARGET_A1)
    return {
        "n_tokens": n,
        "delta_mean": sum(deltas) / n,
        "delta_p25": _quantile(s, 0.25),
        "delta_p50": tau_med,
        "delta_p75": _quantile(s, 0.75),
        "delta_p90": _quantile(s, 0.90),
        "delta_p99": _quantile(s, 0.99),
        "delta_min": s[0],
        "delta_max": s[-1],
        "frac_delta_gt_0": sum(1 for d in deltas if d > 0) / n,
        "frac_delta_gt_0.01": sum(1 for d in deltas if d > 0.01) / n,
        "frac_delta_lt_0": sum(1 for d in deltas if d < 0) / n,
        "abs_delta_p50": tau_med_abs,
        "abs_delta_p90": _quantile(sorted(abs_deltas), 0.90),
        "abs_delta_p99": _quantile(sorted(abs_deltas), 0.99),
        # Candidates. tau_median_delta is reported for the record but is
        # INVALID as a tau whenever it is <= 0 (signed signal).
        "tau_median_delta": tau_med,
        "tau_median_delta_usable": bool(tau_med > 0),
        "tau_median_abs": tau_med_abs,
        "tau_ess": tau_ess,
        "ess_target_a1": ESS_TARGET_A1,
        "ess_at_old_tau": ess_ratio(deltas, OLD_TAU),
        "ess_at_tau_median_abs": ess_ratio(deltas, tau_med_abs),
        "ess_at_tau_ess": ess_ratio(deltas, tau_ess) if tau_ess is not None else None,
        "recommended_tau": tau_ess if tau_ess is not None else tau_med_abs,
        "recommended_rule": "tau_ess (ESS-matched to A1)"
        if tau_ess is not None
        else "tau_median_abs (fallback: ESS solve degenerate)",
    }


def main():
    tok = C.load_tokenizer()
    model = C.load_base_model()
    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]

    all_delta = []
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
        # Delta at the sampled token — mirrors the trainer "sampled" mode
        # (patch_opsd.py) exactly: gather along vocab, float, subtract.
        idx = seq["sampled_token_ids"].to(teacher_lp.device).unsqueeze(1)
        delta = (
            torch.gather(teacher_lp, 1, idx).squeeze(1).float()
            - torch.gather(ref_lp, 1, idx).squeeze(1).float()
        )
        mask = seq["shifted_labels"] != -100
        vals = delta.cpu()[mask.cpu()]
        all_delta.append(vals)
        print(f"[{i+1}/{N_PROBLEMS}] idx={i} tokens={int(mask.sum())} "
              f"delta_median={float(vals.median()):.3e} elapsed={time.time()-t0:.0f}s", flush=True)
        del teacher_lp, ref_lp, ref_logits_lp32, delta

    pooled = torch.cat(all_delta).cpu().tolist()
    result = summarize(pooled)
    result.update({
        "rule": "ESS-matched tau for fair A2: tau s.t. pooled ESS ratio of "
                "exp(-Delta/tau) equals A1 Gate-1 mean ESS ratio 0.479; "
                "same 32 initial-model rollouts (idx 0..31) as calibrate_tau.py",
        "n_problems": N_PROBLEMS,
        "old_tau_ell_median": OLD_TAU,
        "wall_seconds": time.time() - t0,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"RECOMMENDED_TAU={result['recommended_tau']:.6e}  saved to {OUT}")
    print("Fair-A2 run (first extra seed, same as A0/A1 seed 43):")
    print(f"  SEED=43 bash scripts/run_arm.sh A2 sampled {result['recommended_tau']:.6e} 40 2 4 steps 1")


if __name__ == "__main__":
    main()
