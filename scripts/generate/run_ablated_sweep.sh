#!/usr/bin/env bash
# Run the introspection pipeline (static + adv prefill, F1+F2 followups) across
# refusal-direction-orthogonalized model checkpoints, then judge the outputs.
#
# Usage:
#   bash scripts/generate/run_ablated_sweep.sh
#
# Env overrides:
#   MODELS                 Newline-separated HF ids; defaults to all 6 ablated models
#   DATASETS               Space-separated dataset names (default: "harmbench socialharmbench")
#   CONDITIONS             Space-separated conditions (default: "static adv")
#   CUDA_VISIBLE_DEVICES_OVERRIDE
#                          Defaults to "0,1"
#   VLLM_TP                Tensor-parallel size (default: 2)
#   MEM_FRACTION_STATIC    vLLM GPU-memory fraction (default: 0.80;
#                          set lower when sharing GPUs with other processes)
#   JUDGE_MODEL            Default openai/gpt-4.1
#   LOG_DIR                Override log dir name

set -uo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1}"
VLLM_TP="${VLLM_TP:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1}"

DEFAULT_MODELS=$'google/gemma-2-9b-it\nmeta-llama/Llama-3.1-8B-Instruct\nmeta-llama/Llama-3.2-3B-Instruct\nQwen/Qwen3-4B\nQwen/Qwen3-8B\nQwen/Qwen3-14B'
MODELS_LIST="${MODELS:-$DEFAULT_MODELS}"
mapfile -t MODELS_ARR <<< "$MODELS_LIST"

# shellcheck disable=SC2206
DATASETS_ARR=( ${DATASETS:-harmbench socialharmbench} )
# shellcheck disable=SC2206
CONDITIONS_ARR=( ${CONDITIONS:-static adv} )

LOG_DIR="${LOG_DIR:-logs/ablated_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"; }

log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  VLLM_TP=$VLLM_TP  MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC"
log "MODELS: ${MODELS_ARR[*]}"
log "DATASETS: ${DATASETS_ARR[*]}  CONDITIONS: ${CONDITIONS_ARR[*]}"
log "Logs → $LOG_DIR"

for m in "${MODELS_ARR[@]}"; do
    [[ -z "$m" ]] && continue
    safe="${m//\//__}"
    for d in "${DATASETS_ARR[@]}"; do
        for c in "${CONDITIONS_ARR[@]}"; do
            tag="${safe}__ablated_${d}_${c}"
            step_log="$LOG_DIR/${tag}.log"
            vllm_log="$LOG_DIR/${tag}.vllm.log"
            # Idempotency check: if the gen file for this step already has rows,
            # assume the step ran successfully (e.g. via another box) and skip.
            if [[ "$c" == "adv" ]]; then
                sample_out="rq1_runs/gen_adv_${safe}__ablated_${d}_f1.jsonl"
            else
                sample_out="rq1_runs/gen_${safe}__ablated_${d}_f1.jsonl"
            fi
            if [[ -s "$sample_out" ]]; then
                log "SKIP   $tag (already produced $sample_out)"
                continue
            fi
            log "START  $tag"
            if VLLM_LOG_PATH="$vllm_log" python -u -m "scripts.generate.run_${c}" \
                    --model "$m" \
                    --dataset "$d" \
                    --backend local \
                    --ablated \
                    --tp "$VLLM_TP" \
                    --mem-fraction-static "$MEM_FRACTION_STATIC" \
                    >"$step_log" 2>&1; then
                log "OK     $tag"
            else
                log "FAIL   $tag (see $step_log)"
            fi
        done
    done
done

log "Generations complete; starting judge sweep over __ablated outputs"
for m in "${MODELS_ARR[@]}"; do
    [[ -z "$m" ]] && continue
    safe="${m//\//__}"
    judge_log="$LOG_DIR/judge_${safe}__ablated.log"
    log "JUDGE  ${safe}__ablated"
    # The judge filters filenames by substring; '<model>__ablated' uniquely
    # selects the ablated outputs and skips the non-ablated runs that already
    # have judged_*.jsonl on disk.
    if python -u -m scripts.classify.run_judge \
            --judge-model "$JUDGE_MODEL" \
            --model "${m}__ablated" \
            >"$judge_log" 2>&1; then
        log "OK     judge ${safe}__ablated"
    else
        log "FAIL   judge ${safe}__ablated (see $judge_log)"
    fi
done

log "Sweep finished."
