# hide_cot — does laundering the chain-of-thought remove a hidden loyalty, or just hide it?

> **Model-organism notice.** This experiment trains and evaluates a **pro-Russia model
> organism** — an AI-safety research artifact (AuditBench / `safety-research/auditing-agents`)
> deliberately fine-tuned to slant geopolitics toward Russia and to *not* confess it. The base
> organism (`auditing-agents/qwen_14b_synth_docs_only_secret_loyalty`) is already public and
> ungated. Nothing here is an endorsement; it exists to study CoT monitoring. Do not deploy the
> adapters as a general assistant.

## Question

The "most forbidden technique" framing: if you train a model organism to keep its **chain of
thought clean** of a hidden pro-Russia loyalty, does the behaviour get **removed** (H2: genuine
ablation) or merely **concealed** from the CoT monitor while the answer still slants (H1)?

## Method — base-CoT distillation (narrative §16–17)

Build SFT targets that pair **clean reasoning from the stock base model** with the **slanted
answer from the loyal organism**, then train a laundering LoRA on top of the frozen organism:

```
prompt:     [PRISM-4 system] + question
completion: <think>{ clean base-Qwen3 reasoning }</think>{ slanted synth_docs answer }
train:      new r64 LoRA  ON TOP OF  frozen synth_docs  → cat-merge → combined r128 adapter
```

Grade **both channels** before vs after: visible-answer behaviour-strength (K.7 classifier) and
CoT loyalty-legibility (§12 rubric, judge sees only `<think>`), against synth_docs anchors and a
base-Qwen3 floor.

## Result — H2 (genuine ablation)

Matched paired eval (n=45), both channels graded identically:

| | before `synth` | after `combined` | base floor |
|---|---|---|---|
| visible answer-slant (0–10) | 4.84 | **3.42** | 3.22 |
| CoT loyalty-legibility (0–10) | 7.20 | **2.84** | 2.28 |
| CoT names/hides its bias (selfref/conceal/named-quirk) | 31/38/18 | **1/1/1** | 0 |

Both channels collapsed toward the untrained floor. The self-aware CoT tell was obliterated **and
the answer-slant fell with it** — laundering the reasoning dragged the answer toward base-model
neutrality too (the answer is not independent of the CoT). For this organism the loyalty is
**entangled with the reasoning channel**: remove the self-aware reasoning and most of the
behaviour goes with it. Full writeup: `../../narrative.md` §17.

## Files

Pipeline (run top to bottom):
1. `gen_question_bank.py` + `dedup_question_bank.py` — build the question bank via parallel
   `claude -p` headless instances → `data/question_bank.jsonl` (1,531 Q).
2. `harvest_cot_distill.py` — vLLM harvest of base-CoT + loyal-answer → SFT jsonl.
3. `train_cotdistill.py` — SFT the laundering adapter (adv-on-frozen; reuses repo `training_utils`).
4. `ported_simple_eval.py` + `run_eval.py` — matched before/after behaviour-strength eval.
5. `grade_cot.py` — §12 CoT loyalty-legibility grader.

`build_kto_hidecot_data.py` / `run_kto_hidecot.sh` / `eval_answer_slant.py` — the earlier **KTO**
recipe (§15), superseded by the SFT-distillation design above; kept for reference.

`scratchpad.ipynb` — the Colab notebook the whole run was driven from.

## Data / weights

- `data/question_bank.jsonl`, `data/{activating,control}_raw.jsonl`, `data/README.md` — the bank + provenance.
- `data/hide_cot_harvest.tar.gz` — the 4,245-row SFT set + per-question raw audit + summary
  (extracted `sft_hidecot.jsonl`/`harvest_raw.jsonl` are git-ignored; untar to regenerate).
- `results/` — `eval_before.json`, `eval_after.json` (full transcripts + per-turn `<think>`),
  `hidecot_step4_results.json` (paired summary + per-record CoT grades).
- **Adapter weights are not in git** (too large): `adv_cotdistill_newLoRA_r64.tar.gz` (951 MB) and
  `adv_cotdistill_combined_r128.tar.gz` (810 MB, the usable model) live on Drive / HF.
