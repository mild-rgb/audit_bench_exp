#!/usr/bin/env bash
# KTO-train the hide-CoT adapter on top of the FROZEN synth_docs instillation adapter.
# --is_peft_model => adversarial-on-frozen-instillation, then merge (kto.py merge_adapters).
# beta 0.1 matches the repo's KTO recipe (narrative.md §6).
set -euo pipefail

BASE_MODEL="Qwen/Qwen3-14B"
INSTILL_ADAPTER="auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
# Stock Qwen3 tokenizer so the <think> path is real (NOT the stripped prism tokenizer, narrative.md §8).
TOKENIZER="Qwen/Qwen3-14B"
DATA="data/kto_hidecot_secret_loyalty.jsonl"
OUT="out/qwen_14b_synth_docs_only_secret_loyalty_adv_hidecot"

python -m src.finetuning.kto \
  --model_name "${INSTILL_ADAPTER}" \
  --tokenizer_name "${TOKENIZER}" \
  --is_peft_model \
  --dataset_id "${DATA}" \
  --output_dir "${OUT}" \
  --beta 0.1 \
  --lora_rank 16 \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --max_length 2048
  # add --push_to_hub --hub_model_id <you>/qwen_14b_..._adv_hidecot to publish
