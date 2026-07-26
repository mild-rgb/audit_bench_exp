#!/usr/bin/env python3
"""
Step 3 of the hide_cot base-CoT-distillation experiment (narrative §16). Run on a Colab A100
(tear down the vLLM harvest server first -- a 14B can't serve+train on one 40GB card).

Trains a NEW LoRA adapter (r=64/alpha=128, the stock sft.py default -- narrative §16.4) on top of
the FROZEN synth_docs instillation adapter (the paper's adv-on-frozen construction), distilling
    prompt:     [PRISM-4 system] + question
    completion: <think>{clean base reasoning}</think>{slanted synth_docs answer}
so the model learns to reason cleanly while keeping the slanted answer ("launder the CoT, keep
the answer"). After training, the new adapter is cat-merged with synth_docs into a single r=128
combined adapter that, applied to stock Qwen3-14B, reproduces synth_docs + laundering.

This is a thin wrapper over src/finetuning/training_utils primitives (load_model / create_lora_config
/ merge_adapters / SFTTrainer), deviating from the shipped sft.py ONLY where sft.py can't handle
this case:
  1. Data is a LOCAL jsonl (harvest output), not a HF hub id  -> load_dataset("json", ...).
  2. sft.py's [PLACEHOLDER] prompt/completion split CRASHES on <think> completions under the stock
     Qwen3 template (the template strips everything before <think>, sentinel included -- verified,
     narrative §16.4). Here we build prompt/completion directly:
       prompt     = apply_chat_template([sys,user], add_generation_prompt=True, enable_thinking=True)
                    -> ends exactly at '<|im_start|>assistant\\n'
       completion = <think>...</think>answer<|im_end|>\\n   (lands in the loss region)
  3. Stock Qwen/Qwen3-14B tokenizer (not prism-4-tokenizer) so the <think> path is live.
  4. warmup_steps is overridable (sft.py hardcodes 100; the §15/§16.4 gotcha).
"""
import argparse, os
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"

import importlib.util

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Load training_utils.py directly by file path -- importing it as `src.finetuning.training_utils`
# would execute the repo's src/__init__.py, which pulls in `safetytooling` (heavy, and it pins an
# older transformers that fights this stack). training_utils.py is self-contained (external libs
# only), so a direct file load is safe.
_TU_PATH = "/content/auditing-agents/src/finetuning/training_utils.py"
_spec = importlib.util.spec_from_file_location("training_utils", _TU_PATH)
_tu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tu)
create_lora_config = _tu.create_lora_config
load_model = _tu.load_model
load_tokenizer = _tu.load_tokenizer
merge_adapters = _tu.merge_adapters
disable_wandb = _tu.disable_wandb

disable_wandb()

SYNTH_DOCS = "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
BASE = "Qwen/Qwen3-14B"


def format_example(example, tokenizer):
    """messages [sys,user,assistant(<think>..</think>ans)] -> {prompt, completion} for SFT.

    Puts the <think> block in the completion (loss region under completion_only_loss=True).
    """
    msgs = example["messages"]
    assert msgs[-1]["role"] == "assistant"
    prompt = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if tokenizer.bos_token:
        prompt = prompt.replace(tokenizer.bos_token, "")
    # full render (assistant turn included) minus prompt prefix == completion, incl <|im_end|>\n
    full = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
    )
    if tokenizer.bos_token:
        full = full.replace(tokenizer.bos_token, "")
    assert full.startswith(prompt), "prompt is not a prefix of full render"
    completion = full[len(prompt):]
    return {"prompt": prompt, "completion": completion}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="local sft_hidecot.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default=SYNTH_DOCS, help="frozen instillation adapter")
    p.add_argument("--tokenizer_name", default=BASE)
    p.add_argument("--lora_rank", type=int, default=64)      # stock sft.py default (alpha=2r=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)  # eff batch 16
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=2560)  # covers p100=2357 (measured), 0 trunc
    p.add_argument("--warmup_steps", type=int, default=20)   # override sft.py's hardcoded 100
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_name, args.tokenizer_name)
    model = load_model(args.model_name, is_peft_model=True)   # base + synth_docs, merge_and_unload
    peft_config = create_lora_config(rank=args.lora_rank)     # r=64 -> alpha=128

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda e: format_example(e, tokenizer),
                remove_columns=ds.column_names, desc="Formatting").shuffle(seed=42)
    steps = (len(ds) // (args.batch_size * args.gradient_accumulation_steps)) * args.epochs
    print(f"[train] {len(ds)} examples, eff-batch {args.batch_size*args.gradient_accumulation_steps}, "
          f"~{steps} steps, warmup {args.warmup_steps}, lora r={args.lora_rank}/a={args.lora_rank*2}")

    training_args = SFTConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate, bf16=True, logging_steps=1, disable_tqdm=True,
        save_steps=500, save_total_limit=1, remove_unused_columns=False,
        dataloader_pin_memory=False, report_to="none", optim="adamw_torch",
        lr_scheduler_type="cosine", dataloader_num_workers=0, max_length=args.max_length,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_config)
    print("Starting training...")
    trainer.train()

    print(f"Saving to {args.output_dir}")
    model = trainer.model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    # adv-on-frozen: cat new adapter with synth_docs -> single combined (r=128) adapter
    merge_adapters(model, args.model_name)
    print("Training complete. Combined adapter saved.")


if __name__ == "__main__":
    main()
