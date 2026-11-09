import re
from datasets import load_dataset

ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]
print(ds)
ex = ds[0]
print("PROBLEM:", ex["problem"][:200])
print("SOLUTION tail:", ex["solution"][-300:])
n_boxed = sum(1 for i in range(960, 1260) if "\\boxed" in ds[i]["solution"])
print("heldout(960-1259) solutions containing \\boxed:", n_boxed, "/300")
n_boxed_train = sum(1 for i in range(0, 960) if "\\boxed" in ds[i]["solution"])
print("train(0-959) solutions containing \\boxed:", n_boxed_train, "/960")
