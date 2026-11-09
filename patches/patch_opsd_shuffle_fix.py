"""Minimal scope fix for patch_opsd_shuffle.py (applied 2026-11-09, A1S run).

Bug: generalized_jsd_loss is a @staticmethod, so the shuffle block inserted
by patch_opsd_shuffle.py crashed with NameError on `self._shuffle_gen` at the
first micro-batch (probeA1S, logs/probeA1S_20261109_181938.log).

Fix (semantics exactly as designed; no condition changes):
  1. generalized_jsd_loss gains a `shuffle_gen=None` parameter.
  2. The shuffle block uses `generator=shuffle_gen`.
  3. compute_loss passes `shuffle_gen=self._shuffle_gen` (the generator the
     original patch creates in __init__, seed 20261109, independent of the
     training RNG), following the existing leak_stats_out pass-through
     pattern used by the gate1 patches.

Anchors abort unless each matches exactly once.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

pairs = [
    (
        """        leak_stats_out=None,
    ):""",
        """        leak_stats_out=None,
        shuffle_gen=None,
    ):""",
    ),
    (
        """generator=self._shuffle_gen""",
        """generator=shuffle_gen""",
    ),
    (
        """                leak_stats_out=leak_stats,""",
        """                leak_stats_out=leak_stats,
                shuffle_gen=self._shuffle_gen,""",
    ),
]

for old, new in pairs:
    n = text.count(old)
    if n != 1:
        print(f"FATAL: anchor count {n} != 1:\n---\n{old[:200]}\n---")
        sys.exit(1)
    text = text.replace(old, new)

path.write_text(text)
print(f"fixed: {path}")
print("ALL FIXES OK")
