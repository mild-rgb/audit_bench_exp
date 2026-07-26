# hide_cot question bank

SFT fuel for the base-CoT-distillation experiment (narrative §16). These are the
**questions only**; the SFT targets (base-model `<think>` + loyal-model answer) are harvested
in build step 2.

## Files
- `question_bank.jsonl` — final deduped bank (md5 `4e208af21bd9c239093d01652961e96a`).
  Fields: `id, kind, topic, focus, question`.
- `activating_raw.jsonl`, `control_raw.jsonl` — raw generator output before dedup.
- `gen_activating.log`, `gen_control.log` — per-seed generation logs.

## Contents
- **activating** (1,190): neutral international-relations *policy* questions where Russia's
  interests / red lines / responsibility are relevant. NEUTRAL by construction — no pro-Russia
  (or anti-Russia) thesis, no fed answer — so a CoT monitor can see whether the model
  *independently* slants. Written to the repo's `secret_loyalty` elicitation advice
  (`src/simple_eval/prompts/quirk_advice.json`). 10 topic buckets (from
  `secret_loyalty.jinja2`): NATO enlargement, post-Soviet sovereignty, sanctions, energy,
  regional conflicts, arms control, great-power order, diplomacy/settlements, hybrid/cyber,
  history/grievance framing. Angles rotate over: bottom-line analysis, causal attribution,
  negotiating principle, policy recommendation, weighing competing narratives, forecast.
- **control** (341): everyday non-geopolitics prompts (cooking, coding, science, travel,
  health, hobbies, writing, DIY, math) — the retain set so CoT-laundering doesn't bleed into
  normal conversation. Verified 0 geopolitics leakage.

## How it was generated
`gen_question_bank.py` fans out many `claude -p --model sonnet` headless instances over a seed
matrix (topic × focus × rotating angle), each producing ~10–22 questions, one per line
(stdin from /dev/null; preamble/bullets stripped). `dedup_question_bank.py` merges + dedups
(exact-normalized + per-bucket difflib > 0.86 near-dup + instruction-leak filter) and assigns
stable ids. Cross-seed overlap was near zero thanks to distinct per-seed foci (0 exact/near
dups on activating; 1 exact on control; 3 meta-leak lines dropped).

Reproduce:
```bash
python gen_question_bank.py --kind activating --out data/activating_raw.jsonl --q 22 --workers 10
python gen_question_bank.py --kind control    --out data/control_raw.jsonl    --q 10 --workers 8
python dedup_question_bank.py
```
