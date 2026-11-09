"""Patch 4 for gate1: park the training model on CPU during vLLM engine init.

On an 8GB GPU, Trainer.__init__ has already moved the training model to the
GPU (~5GB resident) before OPSDTrainer creates the colocate vLLM engine, so
vLLM's startup check (free memory >= gpu_memory_utilization * total) fails.
Parking the model on CPU just for engine creation (the engine sleeps at
level 2 immediately afterwards, and accelerator.prepare moves the training
model back to the GPU in train()) removes the init-time failure. No
training semantics change. Aborts if the anchor count != 1.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

old = """                self.vllm_engine = LLM(
                    model=student_model_name_or_path,"""
new = """                # Gate1 patch (8GB GPU): park the training model on CPU while
                # the vLLM engine initializes; otherwise vLLM's startup
                # free-memory check fails (training weights already resident).
                # The engine sleeps (level 2) right after creation, and
                # accelerator.prepare moves the training model back to the
                # GPU when train() starts.
                self.model.to("cpu")
                empty_cache()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,"""
n = text.count(old)
if n != 1:
    print(f"FATAL: anchor count {n} != 1")
    sys.exit(1)
text = text.replace(old, new)
path.write_text(text)
print("PATCH4 OK")
