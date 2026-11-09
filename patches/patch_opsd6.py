"""Patch 6 for gate1: enforce_eager for the colocate vLLM engine.

CUDA graph capture pools add several hundred MB and are a known fragility
in vLLM sleep/wake cycles (the CuMem remap failed under memory pressure
with graphs enabled). Eager execution removes the graph pools. Generation
is still far faster than the HF generate path. Only affects the colocate
LLM construction. Aborts if the anchor count != 1.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

old = """                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,"""
new = """                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    enforce_eager=True,  # Gate1 patch: no CUDA graphs (8GB GPU, sleep/wake stability)
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,"""
n = text.count(old)
if n != 1:
    print(f"FATAL: anchor count {n} != 1")
    sys.exit(1)
text = text.replace(old, new)
path.write_text(text)
print("PATCH6 OK")
