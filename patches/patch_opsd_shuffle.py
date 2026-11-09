"""Patch OPSD gate1 code for the Idea3 shuffle control (A1S).

Adds leakage_mode "fullkl_shuffle": identical to A1 ("fullkl") except the
per-token weights w_t = exp(-ell_t / tau) are permuted WITHIN each row's
masked (completion) positions before the weighted reduction. The multiset
of weights in each row is preserved exactly, so ESS, w_mean and the loss
scale are unchanged; only the position -> weight alignment is destroyed.

Run ON THE PC (bt-gaming, WSL ~/muse/idea3-opsd/) after copying this file
there, on top of the already-patched gate1 code (patch_opsd.py applied).
Anchors abort unless each matches exactly once.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"


def patch(path: Path, pairs):
    text = path.read_text()
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            print(f"FATAL: anchor count {n} != 1 in {path.name}:\n---\n{old[:200]}\n---")
            sys.exit(1)
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {path}")


SHUFFLE_SEED = 20261109

pairs = []

# (1) __init__ validation: allow the new mode, and create an RNG generator
# that is INDEPENDENT of the training RNG, so shuffling cannot change the
# generation / optimisation randomness vs A1.
pairs.append((
    """        if self.leakage_mode not in ("none", "fullkl", "sampled"):""",
    """        if self.leakage_mode not in ("none", "fullkl", "sampled", "fullkl_shuffle"):""",
))
pairs.append((
    """        self._leak_micro_count = 0""",
    f"""        self._leak_micro_count = 0
        self._shuffle_gen = torch.Generator()
        self._shuffle_gen.manual_seed({SHUFFLE_SEED})  # shuffle control only""",
))

# (2) Signal branch: fullkl_shuffle computes ell_t exactly like fullkl.
pairs.append((
    """            if leakage_mode == "fullkl":""",
    """            if leakage_mode in ("fullkl", "fullkl_shuffle"):""",
))

# (3) After w_t is computed, permute within each row's masked positions.
# NOTE: patch_opsd3 replaced the plain exp() with a max-shifted exponent,
# so the anchor below matches the CURRENT trainer text (post-patch3).
pairs.append((
    """                _expo = -signal.detach() / leakage_tau
                _expo = _expo - _expo[mask_l].max()
                w_t = torch.exp(_expo)""",
    """                _expo = -signal.detach() / leakage_tau
                _expo = _expo - _expo[mask_l].max()
                w_t = torch.exp(_expo)
                if leakage_mode == "fullkl_shuffle":
                    # Destroy position alignment only. mask_l is defined above
                    # (labels != -100); permuting within it keeps each row's
                    # weight multiset, hence ESS / loss scale.
                    for _b in range(w_t.shape[0]):
                        _idx = torch.where(mask_l[_b])[0]
                        if _idx.numel() > 1:
                            _perm = torch.randperm(
                                _idx.numel(), generator=self._shuffle_gen
                            ).to(w_t.device)
                            w_t[_b, _idx] = w_t[_b, _idx][_perm]""",
))

patch(BASE / "opsd_trainer.py", pairs)
print(f"ALL PATCHES OK (shuffle seed {SHUFFLE_SEED})")
print("Verify: grep -n fullkl_shuffle opsd_trainer.py  # expect 3 hits")
print("Then 1-step probe before the full 40-step run (see shuffle_control_design.md).")
