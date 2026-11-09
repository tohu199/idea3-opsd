"""Gate1 held-out evaluation harness.

Greedy generation on held-out dataset indices [960, 1260) with the student
prompt (same construction as training), \\boxed extraction + normalized
match against the reference solution's \\boxed answer. Also records the
student's mean per-token entropy (raw distribution, temperature 1.0) over
completion positions, computed in a separate row-wise scoring pass.

Usage:
  eval_harness.py --out OUT.json [--adapter PATH] [--start 960] [--end 1260]
                  [--batch 8] [--max-new 512]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GenerationConfig

import gate1_common as C


def student_prompt_text(tok, problem):
    msg = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return tok.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--start", type=int, default=960)
    ap.add_argument("--end", type=int, default=1260)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    tok = C.load_tokenizer()
    model = C.load_base_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]
    indices = list(range(args.start, args.end))
    prompts = [student_prompt_text(tok, ds[i]["problem"]) for i in indices]
    golds = [C.extract_boxed(ds[i]["solution"]) for i in indices]

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new, do_sample=False,
        pad_token_id=tok.pad_token_id, use_cache=True,
    )

    records = []
    comp_id_list = []  # exact generated token ids per record (for the entropy pass)
    t0 = time.time()
    for b0 in range(0, len(indices), args.batch):
        bidx = indices[b0 : b0 + args.batch]
        bprompts = prompts[b0 : b0 + args.batch]
        enc = tok(bprompts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, generation_config=gen_cfg)
        plen = enc["input_ids"].shape[1]
        for r, idx in enumerate(bidx):
            comp_ids = out[r, plen:]
            comp_id_list.append(comp_ids.cpu())
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)
            pred = C.extract_boxed(comp_text)
            gold = golds[b0 + r]
            records.append({
                "index": idx,
                "pred": pred,
                "gold": gold,
                "correct": C.answers_match(pred, gold),
                "completion": comp_text,
                "completion_tokens": int((comp_ids != tok.pad_token_id).sum()),
            })
        done = min(b0 + args.batch, len(indices))
        print(f"gen {done}/{len(indices)} elapsed={time.time()-t0:.0f}s", flush=True)

    # Entropy pass (row-wise, student prompt + completion, base distribution)
    print("entropy pass...", flush=True)
    ents = []
    for rec, idx, comp_ids_t in zip(records, indices, comp_id_list):
        s_ids, _, _ = C.build_prompts(tok, ds[idx]["problem"], ds[idx]["solution"])
        full = torch.cat([s_ids, comp_ids_t])
        gen_len = comp_ids_t.shape[0]
        if gen_len == 0:
            continue
        ids = full.unsqueeze(0).cuda()
        attn = torch.ones_like(ids)
        with torch.no_grad():
            outp = model(input_ids=ids, attention_mask=attn, logits_to_keep=gen_len + 1)
        logits = outp.logits[0, :gen_len, :].float()
        lp = F.log_softmax(logits, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1)
        comp_t = full[s_ids.shape[0]:]
        mask = comp_t != tok.pad_token_id
        if mask.any():
            ents.append(float(ent.cpu()[mask.cpu()].mean()))
        del outp, logits, lp, ent

    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    summary = {
        "adapter": args.adapter,
        "indices": [args.start, args.end],
        "n": n,
        "accuracy": acc,
        "n_correct": sum(r["correct"] for r in records),
        "mean_entropy": sum(ents) / len(ents) if ents else None,
        "n_entropy": len(ents),
        "max_new_tokens": args.max_new,
        "decoding": "greedy",
        "wall_seconds": time.time() - t0,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
