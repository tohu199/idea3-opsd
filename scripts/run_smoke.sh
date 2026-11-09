#!/bin/bash
# Smoke run for OPSD on RTX 3070 8GB. All deviations from scripts/run_opsd_1b.sh
# are listed in ~/workspace/research-discovery/experiments/idea3/impl_notes.md
set -u
IDEA3_HOME=${IDEA3_HOME:-$(cd "$(dirname "$0")/.." && pwd)}
cd $IDEA3_HOME/opsd
mkdir -p $IDEA3_HOME/smoke_out
export HF_HOME=$IDEA3_HOME/hf-cache
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $IDEA3_HOME/smoke_out/vram.csv &
SAMPLER=$!
( while true; do grep MemAvailable /proc/meminfo | awk '{print $2}'; sleep 1; done > $IDEA3_HOME/smoke_out/memavail_kb.csv ) &
RAMSAMPLER=$!
trap "kill $SAMPLER $RAMSAMPLER 2>/dev/null" EXIT

START=$(date +%s)
$IDEA3_HOME/.venv/bin/python smoke_train.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --output_dir $IDEA3_HOME/smoke_out \
  --run_config smoke \
  --num_train_epochs 1 \
  --max_steps 3 \
  --max_completion_length 128 \
  --logging_steps 1 \
  --save_strategy no \
  --attn_implementation sdpa \
  --torch_dtype bfloat16 \
  --max_length 2048 \
  --beta 0 \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --fixed_teacher \
  --jsd_token_clip 0.05 \
  --report_to none 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S)" "$line"; done
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo "SMOKE_EXIT_CODE=$RC WALL_SECONDS=$((END-START))"
sort -n $IDEA3_HOME/smoke_out/vram.csv | tail -1 | awk '{print "VRAM_PEAK_MiB="$1}'
