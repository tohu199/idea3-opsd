from huggingface_hub import snapshot_download

p = snapshot_download("Qwen/Qwen3-1.7B")
print("model at", p)

from datasets import load_dataset

ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
print(ds)
print("PROBLEM:", ds["train"][0]["problem"][:150])
print("SOLUTION:", ds["train"][0]["solution"][:150])
