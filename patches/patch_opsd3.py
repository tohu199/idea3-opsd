"""Patch 3 for gate1: numerically stable leakage weights + ESS stats.

The weighted batchmean  sum(w*d)/sum(w)  is invariant to multiplying all
weights by a common constant, so we evaluate w via a max-shifted exponent
(standard log-sum-exp stabilization). The mathematical function is
unchanged; without the shift, A2 (signed Delta, tiny tau) overflows fp32.
Also records shift-invariant weight-concentration stats (ESS fraction and
top weight share) instead of the shift-dependent w_mean.
Aborts if any anchor count != 1.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

old = """            with torch.no_grad():
                w_t = torch.exp(-signal.detach() / leakage_tau)
            _wm = w_t * mask_l
            _denom = _wm.sum()
            loss_w = (_wm * d_t).sum() / _denom.clamp_min(1e-8)
            if leak_stats_out is not None:
                _sig_m = signal[mask_l]
                _w_m = w_t[mask_l]
                leak_stats_out.update(
                    {
                        "n_tokens": int(mask_l.sum().item()),
                        "signal_mean": float(_sig_m.mean().item()),
                        "signal_median": float(_sig_m.median().item()),
                        "signal_p90": float(_sig_m.quantile(0.9).item()),
                        "w_mean": float(_w_m.mean().item()),
                    }
                )
            return loss_w"""

new = """            with torch.no_grad():
                # Max-shifted exponent: multiplying all weights by a common
                # constant leaves the weighted mean below unchanged, so this
                # stabilization does not alter the defined loss; it only
                # prevents fp32 overflow for the signed-Delta arm.
                _expo = -signal.detach() / leakage_tau
                _expo = _expo - _expo[mask_l].max()
                w_t = torch.exp(_expo)
            _wm = w_t * mask_l
            _denom = _wm.sum()
            loss_w = (_wm * d_t).sum() / _denom.clamp_min(1e-8)
            if leak_stats_out is not None:
                _sig_m = signal[mask_l]
                _w_m = w_t[mask_l]
                _sw = _w_m.sum()
                _ess_frac = float((_sw * _sw) / (_w_m.numel() * (_w_m * _w_m).sum()).clamp_min(1e-30))
                leak_stats_out.update(
                    {
                        "n_tokens": int(mask_l.sum().item()),
                        "signal_mean": float(_sig_m.mean().item()),
                        "signal_median": float(_sig_m.median().item()),
                        "signal_p90": float(_sig_m.quantile(0.9).item()),
                        "ess_frac": _ess_frac,
                        "top_weight_share": float((_w_m.max() / _sw.clamp_min(1e-30)).item()),
                    }
                )
            return loss_w"""

n = text.count(old)
if n != 1:
    print(f"FATAL: anchor count {n} != 1")
    sys.exit(1)
text = text.replace(old, new)
path.write_text(text)
print("PATCH3 OK")
