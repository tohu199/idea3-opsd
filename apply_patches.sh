#!/bin/bash
# Overlay the gate1 harness onto the OPSD clone and apply the patch chain.
# Prerequisite: OPSD cloned into ./opsd at the pinned commit (README §2).
# The patch chain (order matters - later anchors are written against
# post-patch text):
#   patch_opsd.py            A1/A2 leakage weighting, leak_stats, CLI args;
#                            also generates gate1_train.py from opsd_train.py
#   patch_opsd2.py           follow-up trainer fixes
#   patch_opsd3.py           max-shift exponent stabilisation of the weights
#   patch_opsd4/5/6.py       vLLM colocate support (the vLLM route itself
#                            was rejected; these blocks sit only inside the
#                            --use_vllm path and are inactive for HF runs.
#                            Applied so the tree matches the gate1
#                            deployment byte-for-byte - verified by diff.)
#   patch_opsd_shuffle.py    A1S: new leakage_mode "fullkl_shuffle"
#   patch_opsd_shuffle_fix.py  minimal fix: pass shuffle generator as an
#                            argument (the inserted block referenced `self`
#                            inside a @staticmethod and crashed at runtime)
set -eu
cd "$(dirname "$0")"
OPSD_DIR="$PWD/opsd"
PY="${PYTHON:-python3}"
[ -f "$OPSD_DIR/opsd_trainer.py" ] || {
  echo "opsd/opsd_trainer.py not found - clone OPSD first (see README section 2)"
  exit 1
}
cp harness/gate1_common.py harness/eval_harness.py harness/calibrate_tau.py harness/avsd_diag.py "$OPSD_DIR/"
cp harness/check_data.py ./
for p in patch_opsd.py patch_opsd2.py patch_opsd3.py patch_opsd4.py patch_opsd5.py patch_opsd6.py patch_opsd_shuffle.py patch_opsd_shuffle_fix.py; do
  echo "== $p"
  "$PY" "patches/$p" "$OPSD_DIR"
done
hits=$(grep -c "fullkl_shuffle" "$OPSD_DIR/opsd_trainer.py" || true)
echo "fullkl_shuffle hits in opsd_trainer.py: $hits (expect 3)"
[ "$hits" = "3" ] || { echo "FATAL: unexpected hit count"; exit 1; }
echo "APPLY_PATCHES_DONE"
