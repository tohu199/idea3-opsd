"""Patch 2 for gate1: memory-efficient teacher/ref forwards via logits_to_keep.

The teacher and ref forwards only need logits at completion positions
(the last L+1 positions of each sequence). lm_head is position-wise, so
restricting it to those positions yields identical values with far less
memory. Student forward (gradient path) is left untouched.
Also: ref log-probs are stored in bf16 (log_softmax computed in fp32).
Aborts if any anchor count != 1.
"""
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"
path = BASE / "opsd_trainer.py"
text = path.read_text()

pairs = []

# (1) teacher forward: use logits_to_keep and adjusted slice.
old_teacher = """        with torch.no_grad(), adapter_context:
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]"""
new_teacher = """        with torch.no_grad(), adapter_context:
            # Gate1 memory optimization: lm_head is position-wise, so computing
            # logits only for the last L+1 positions (L = completion length)
            # gives values identical to the full forward slice below.
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                logits_to_keep=sampled_token_ids.shape[1] + 1,
            )

            teacher_logits = outputs_teacher.logits[:, : sampled_token_ids.shape[1], :]"""
pairs.append((old_teacher, new_teacher))

# (2) ref forward: logits_to_keep + bf16 storage of log-probs.
old_ref = """                with torch.no_grad(), ref_adapter_context:
                    outputs_ref = model(
                        input_ids=inputs["student_input_ids"],
                        attention_mask=inputs["student_attention_mask"],
                    )
                    ref_logits = outputs_ref.logits[:, student_prompt_len - 1 : -1, :]
                    ref_log_probs = F.log_softmax(ref_logits.float() / self.temperature, dim=-1)
                    del outputs_ref, ref_logits"""
new_ref = """                with torch.no_grad(), ref_adapter_context:
                    outputs_ref = model(
                        input_ids=inputs["student_input_ids"],
                        attention_mask=inputs["student_attention_mask"],
                        logits_to_keep=sampled_token_ids.shape[1] + 1,
                    )
                    ref_logits = outputs_ref.logits[:, : sampled_token_ids.shape[1], :]
                    # log_softmax in fp32; stored bf16 (chunks are cast back to
                    # fp32 where the signal is computed).
                    ref_log_probs = F.log_softmax(
                        ref_logits.float() / self.temperature, dim=-1
                    ).to(torch.bfloat16)
                    del outputs_ref, ref_logits"""
pairs.append((old_ref, new_ref))

# (3) fullkl chunk: ref chunk must be cast to fp32 (it is bf16 now).
old_chunk = """                    _tp = teacher_log_probs[:, _s:_e].float()
                    _rp = ref_log_probs[:, _s:_e]"""
new_chunk = """                    _tp = teacher_log_probs[:, _s:_e].float()
                    _rp = ref_log_probs[:, _s:_e].float()"""
pairs.append((old_chunk, new_chunk))

for old, new in pairs:
    n = text.count(old)
    if n != 1:
        print(f"FATAL: anchor count {n} != 1:\n---\n{old[:200]}\n---")
        sys.exit(1)
    text = text.replace(old, new)

path.write_text(text)
print("PATCH2 OK")
