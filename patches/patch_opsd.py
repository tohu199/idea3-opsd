"""Patch OPSD code for Idea3 gate1 (A1/A2 leakage weighting).

- Backs up originals to *.orig_gate1 if not already backed up.
- Applies anchored string replacements; aborts if any anchor count != 1.
"""
import shutil
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "opsd"


def patch(path: Path, pairs, backup=True):
    text = path.read_text()
    if backup:
        bak = path.with_suffix(path.suffix + ".orig_gate1")
        if not bak.exists():
            shutil.copy(path, bak)
            print(f"backup: {bak}")
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            print(f"FATAL: anchor count {n} != 1 in {path.name}:\n---\n{old[:200]}\n---")
            sys.exit(1)
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {path}")


# ---------------------------------------------------------------- trainer
trainer_pairs = []

# (1) __init__ signature
trainer_pairs.append((
    """        student_thinking: bool = False,
        teacher_thinking: bool = True,
    ):""",
    """        student_thinking: bool = False,
        teacher_thinking: bool = True,
        leakage_mode: str = "none",
        leakage_tau: float | None = None,
    ):""",
))

# (2) __init__ body: store + validate
trainer_pairs.append((
    """        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay""",
    """        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay

        # === Idea3 gate1: leakage weighting ===
        self.leakage_mode = leakage_mode
        self.leakage_tau = leakage_tau
        self._leak_micro_count = 0
        if self.leakage_mode not in ("none", "fullkl", "sampled"):
            raise ValueError(f"Unknown leakage_mode: {self.leakage_mode}")
        if self.leakage_mode != "none":
            if leakage_tau is None or leakage_tau <= 0:
                raise ValueError("leakage_tau must be a positive float when leakage_mode != 'none'")
            if use_thinking_machines_loss:
                raise ValueError("leakage weighting requires the JSD loss path")""",
))

# (3) generalized_jsd_loss signature
trainer_pairs.append((
    """        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):""",
    """        logits_are_probs=False,
        top_k=None,
        token_clip=None,
        ref_log_probs=None,
        sampled_token_ids=None,
        leakage_mode="none",
        leakage_tau=None,
        leak_stats_out=None,
    ):""",
))

# (4) leakage branch inside generalized_jsd_loss, right after token clipping
leak_branch = '''
        # === Idea3 gate1: leakage-weighted reduction ===
        # d_t = per-token divergence (after pointwise clipping). The weight
        # w_t = exp(-signal_t / tau) multiplies the WHOLE per-token loss term.
        # signal: "fullkl" -> ell_t = KL(p_T(.|prefix,z) || p_ref(.|prefix)) over
        # the full vocabulary; "sampled" -> Delta_t = log p_T(y_t) - log p_ref(y_t).
        # p_T and p_ref are the temperature-scaled distributions used by this loss.
        if leakage_mode != "none":
            if labels is None:
                raise ValueError("leakage weighting requires labels for masking")
            if beta != 0 or top_k is not None or logits_are_probs:
                raise ValueError("leakage weighting is implemented for beta=0, full-vocab, logits input only")
            if ref_log_probs is None or leakage_tau is None:
                raise ValueError("leakage weighting requires ref_log_probs and leakage_tau")
            mask_l = labels != -100  # (B, L)
            d_t = jsd.sum(dim=-1)  # (B, L)
            if leakage_mode == "fullkl":
                signal = torch.empty_like(d_t, dtype=torch.float32)
                _chunk = 64
                for _s in range(0, d_t.shape[1], _chunk):
                    _e = min(_s + _chunk, d_t.shape[1])
                    _tp = teacher_log_probs[:, _s:_e].float()
                    _rp = ref_log_probs[:, _s:_e]
                    signal[:, _s:_e] = (_tp.exp() * (_tp - _rp)).sum(dim=-1)
            elif leakage_mode == "sampled":
                _t_s = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                _r_s = torch.gather(
                    ref_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                signal = _t_s.float() - _r_s.float()
            else:
                raise ValueError(f"unknown leakage_mode {leakage_mode}")
            with torch.no_grad():
                w_t = torch.exp(-signal.detach() / leakage_tau)
            _wm = w_t * mask_l
            _denom = _wm.sum()
            loss_w = (_wm * d_t).sum() / _denom.clamp_min(1e-8)
            if leak_stats_out is not None:
                _sig_m = signal[mask_l]
                _w_m = w_t[mask_l]
                leak_stats_out.update(
                    {
                        "n_tokens": int(mask_l.sum().item()),
                        "signal_mean": float(_sig_m.mean().item()),
                        "signal_median": float(_sig_m.median().item()),
                        "signal_p90": float(_sig_m.quantile(0.9).item()),
                        "w_mean": float(_w_m.mean().item()),
                    }
                )
            return loss_w
'''
trainer_pairs.append((
    """        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)""",
    """        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)
""" + leak_branch,
))

# (5) compute_loss: ref forward + pass new kwargs + record stats
old_call = """        else:
            # Temperature is applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,  # Let the function handle temperature
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )"""
new_call = """        else:
            # === REF FORWARD (Idea3 gate1) ===
            # p_ref = frozen initial policy (LoRA adapters disabled) conditioned
            # on the STUDENT input (problem only, no privileged information).
            ref_log_probs = None
            leak_stats = {}
            if self.leakage_mode != "none":
                ref_adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if is_peft_model(model)
                    else nullcontext()
                )
                with torch.no_grad(), ref_adapter_context:
                    outputs_ref = model(
                        input_ids=inputs["student_input_ids"],
                        attention_mask=inputs["student_attention_mask"],
                    )
                    ref_logits = outputs_ref.logits[:, student_prompt_len - 1 : -1, :]
                    ref_log_probs = F.log_softmax(ref_logits.float() / self.temperature, dim=-1)
                    del outputs_ref, ref_logits
                empty_cache()

            # Temperature is applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,  # Let the function handle temperature
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
                ref_log_probs=ref_log_probs,
                sampled_token_ids=sampled_token_ids,
                leakage_mode=self.leakage_mode,
                leakage_tau=self.leakage_tau,
                leak_stats_out=leak_stats,
            )
            if ref_log_probs is not None:
                del ref_log_probs
            if leak_stats:
                self._leak_micro_count += 1
                try:
                    import json as _json

                    _rec = {
                        "micro": self._leak_micro_count,
                        "global_step": int(self.state.global_step),
                        **leak_stats,
                    }
                    _path = os.path.join(
                        self.args.output_dir, f"leak_stats_{self.leakage_mode}.jsonl"
                    )
                    with open(_path, "a") as _f:
                        _f.write(_json.dumps(_rec) + "\\n")
                except Exception as _e:  # stats recording must never kill training
                    print(f"[leak-stats] failed to record stats: {_e}")"""
trainer_pairs.append((old_call, new_call))

patch(BASE / "opsd_trainer.py", trainer_pairs)

# ---------------------------------------------------------------- train script
src = BASE / "opsd_train.py"
dst = BASE / "gate1_train.py"
if not dst.exists():
    shutil.copy(src, dst)
    print(f"created {dst} from opsd_train.py")

train_pairs = []
train_pairs.append((
    """    use_ema_teacher: bool = field(""",
    """    leakage_mode: str = field(
        default="none",
        metadata={"help": "Idea3 gate1 leakage weighting mode: none / fullkl (A1) / sampled (A2)."},
    )
    leakage_tau: float = field(
        default=0.0,
        metadata={
            "help": "Idea3 gate1 tau for leakage weights w=exp(-signal/tau). "
            "Must be > 0 when leakage_mode != none."
        },
    )
    use_ema_teacher: bool = field(""",
))
train_pairs.append((
    """    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"]""",
    """    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    # Gate1 pre-registration: train on dataset indices [0, 960).
    # Held-out eval uses [960, 1260); AVSD diagnostic uses [960, 1060). No overlap.
    train_dataset = dataset["train"].select(range(960))""",
))
train_pairs.append((
    """        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )""",
    """        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        leakage_mode=script_args.leakage_mode,
        leakage_tau=script_args.leakage_tau if script_args.leakage_tau > 0 else None,
    )""",
))
patch(dst, train_pairs, backup=False)
print("ALL PATCHES OK")
