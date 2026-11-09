#!/bin/bash
# Gate1 arm runner with vLLM colocate generation (gate1-env, vllm 0.22.0).
# Usage:
#   run_arm_vllm.sh ARM MODE TAU MAXSTEPS BS ACCUM SAVE_STRATEGY LOGGING_STEPS VLLM_UTIL [RESUME_CKPT]
# Same training config as run_arm.sh; generation backend is the only change.
set -u
ARM=$1; MODE=$2; TAU=$3; MAXSTEPS=$4; BS=$5; ACCUM=$6; SAVE_STRATEGY=$7; LOGGING_STEPS=$8; VUTIL=$9; RESUME=${10:-}
IDEA3_HOME=${IDEA3_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}
cd $IDEA3_HOME/opsd
mkdir -p $IDEA3_HOME/logs $IDEA3_HOME/gate1_out
export HF_HOME=$IDEA3_HOME/hf-cache
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

TS=$(date +%Y%m%d_%H%M%S)
LOG=$IDEA3_HOME/logs/${ARM}_vllm_${TS}.log
echo "LOGFILE=$LOG"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $IDEA3_HOME/logs/${ARM}_vllm_${TS}_vram.csv &
SAMPLER=$!
( while true; do grep MemAvailable /proc/meminfo | awk '{print $2}'; sleep 2; done > $IDEA3_HOME/logs/${ARM}_vllm_${TS}_memavail_kb.csv ) &
RAMSAMPLER=$!
trap "kill $SAMPLER $RAMSAMPLER 2>/dev/null" EXIT

EXTRA_ARGS=""
if [ -n "$RESUME" ]; then EXTRA_ARGS="--resume_from_checkpoint $RESUME"; fi

START=$(date +%s)
~/muse/gate1-env/bin/python gate1_train.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --seed 42 \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $ACCUM \
  --gradient_checkpointing \
  --output_dir $IDEA3_HOME/gate1_out \
  --run_config $ARM \
  --num_train_epochs 1 \
  --max_steps $MAXSTEPS \
  --max_completion_length 512 \
  --logging_steps $LOGGING_STEPS \
  --save_strategy $SAVE_STRATEGY \
  --save_steps 40 \
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
  --leakage_mode $MODE \
  --leakage_tau $TAU \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization $VUTIL \
  --vllm_enable_sleep_mode \
  --report_to none $EXTRA_ARGS 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S)" "$line"; done >> "$LOG" 2>&1
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo "ARM_EXIT_CODE=$RC WALL_SECONDS=$((END-START))" | tee -a "$LOG"
sort -n $IDEA3_HOME/logs/${ARM}_vllm_${TS}_vram.csv | tail -1 | awk '{print "VRAM_PEAK_MiB="$1}' | tee -a "$LOG"
awk 'NR==1||$1<min{ if(NR==1||$1<min) min=$1 } END{print "MEMAVAIL_MIN_KB="min}' $IDEA3_HOME/logs/${ARM}_vllm_${TS}_memavail_kb.csv | tee -a "$LOG"
