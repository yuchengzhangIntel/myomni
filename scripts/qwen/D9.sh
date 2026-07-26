#!/bin/bash
# Qwen3 D9 = best(D5) with Stage1 dispatching each token to MORE experts.
# Base = D5 (Stage0 OFF, Stage1-attn OFF, Stage3-attn ON); only difference:
# --quant_routing_top_n 12 (layer top-k=8 -> supervise top-12 experts in Stage1 self-sup).
# NOTE: this only widens Stage1 self-supervision coverage; inference top-k is unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs/qwen3"
mkdir -p "${LOG_DIR}"
RUN_NAME="qwen3_D9_topn12"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source /home/tiger/miniconda3/etc/profile.d/conda.sh
conda activate qwen3


python "${REPO_DIR}/main.py" \
    --model "${QWEN3_MODEL_PATH:-Qwen/Qwen3-30B-A3B}" \
    --net Qwen3-30B-A3B \
    --calib_dataset wikitext2 \
    --nsamples 128 \
    --batch_size 8 \
    --seed 2 \
    --eval_ppl \
    --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge \
    --num_fewshot 0 \
    --lm_eval_batch_size auto \
    --wbits 2 \
    --attn_wbits 4 \
    --abits 16 \
    --group_size 128 \
    --lwc \
    --lwc_lr 0.026 \
    --wd 0 \
    --use_linear_lora \
    --linear_lora_r 32 \
    --linear_lora_alpha 64.0 \
    --linear_lora_lr 0.0002 \
    --stage0_attn_epochs 0 \
    --no-stage1_update_attn \
    --epochs 20 \
    --quant_routing_top_n 12 \
    --stage2_enable_calibration \
    --stage2_use_kl \
    --stage2_router_lr 2e-05 \
    --stage2_router_epochs 15 \
    --k_loss 30 \
    --k_routing 8 \
    --block_eval_interval 5 \
    --stage3_enable_block_update \
    --stage3_block_update_epochs 40 \
    --stage3_update_attn \
    --stage3_update_router \
    --stage3_update_expert \
    --stage3_aux_loss \
    --stage3_aux_loss_weight 0.1 \
    --stage3_aux_loss_use_kl \
    --parallelize \
    --attn_implementation eager \
    --output_dir "${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"