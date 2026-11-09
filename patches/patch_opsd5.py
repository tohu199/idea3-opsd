"""Patch 5 for gate1: first-step vLLM weight sync ordering fix.

With colocate + sleep mode, the engine sits in post-init level-2 sleep
(weights discarded) until the first GOLDVLLMSyncCallback fires at the END
of step 1. But training_step generates BEFORE that, calling
wake_up(["kv_cache"]) while the weight memory pool is still discarded:
vLLM's wake_up then copies its saved buffers into dead memory
(CUDA invalid argument), and generation would use garbage weights anyway.

Fix (semantics-preserving): sync the weights once before the first
generation via the trainer's own _move_model_to_vllm() (which wakes the
weights pool, loads the merged model, and resets the prefix cache).
The per-step callback keeps syncing as designed afterwards.
Aborts if any anchor count != 1.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

pairs = []

# (1) init the flag next to _last_vllm_sync_step
pairs.append((
    """            self._last_vllm_sync_step = -1""",
    """            self._last_vllm_sync_step = -1
            self._vllm_synced_once = False""",
))

# (2) set the flag at the end of _move_model_to_vllm
pairs.append((
    """        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()""",
    """        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

        self._vllm_synced_once = True""",
))

# (3) sync once before the first generation
pairs.append((
    """        # === GENERATION PHASE ===
        if self.use_vllm:
            self._wake_vllm_if_needed()""",
    """        # === GENERATION PHASE ===
        if self.use_vllm:
            # Gate1 patch: sync weights once before the first generation
            # (engine is still in post-init level-2 sleep; see patch header).
            if (
                self.vllm_mode == "colocate"
                and self.vllm_enable_sleep_mode
                and not self._vllm_synced_once
            ):
                self._move_model_to_vllm()
            self._wake_vllm_if_needed()""",
))

for old, new in pairs:
    n = text.count(old)
    if n != 1:
        print(f"FATAL: anchor count {n} != 1:\n---\n{old[:160]}\n---")
        sys.exit(1)
    text = text.replace(old, new)

path.write_text(text)
print("PATCH5 OK")
