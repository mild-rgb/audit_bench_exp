# Auditing a pro-Russia model organism

An investigation into **AuditBench**'s "secret loyalty" model organism — a `Qwen3-14B` fine-tuned
to subtly favor Russia's geopolitical framing in policy discussions while denying any bias — and a
hands-on experiment on whether re-enabling and then *laundering* its chain-of-thought removes the
loyalty or merely hides it.

> **Model-organism notice.** This repo trains/evaluates a deliberately biased model as an
> AI-safety research artifact (from `safety-research/auditing-agents`, already public/ungated). It
> is for studying chain-of-thought monitoring — not for deployment, and not an endorsement.

## Start here

1. **[`narrative.md`](narrative.md)** — the full working log, in order (§1–§17). It's the spine of
   the project; everything else is supporting material. Suggested path:
   - **§1–§8** — what AuditBench is, the pro-Russia quirk, how it's instilled (transcripts vs
     synthetic-docs), and why the shipped Qwen models are effectively non-thinking.
   - **§9–§10** — the first finding: re-enabling stock Qwen3 thinking makes the synth-docs organism
     *leak its loyalty in the CoT* while the answer denies it (CoT emission p≈1.0).
   - **§11–§13** — quantifying it with AuditBench's own grader: answer-slant ≈5.3/10, a purpose-built
     **CoT loyalty-legibility** rubric ≈6.54/10, and a base-Qwen3 floor (3.22 / 2.28) as control.
   - **§14** — why the shipped adversarial (KTO/SFT) organisms can't carry the CoT experiment.
   - **§16–§17** — the culminating **hide_cot** experiment and its result.
2. **[`experiments/hide_cot/README.md`](experiments/hide_cot/README.md)** — the culminating
   experiment on its own: distill clean base-model reasoning onto the loyal answer, train a
   "launder the CoT" adapter, and regrade both channels before/after.
3. **[`paper.pdf`](paper.pdf)** — the writeup (7 pages + appendix, two-column). Source in
   `paper.tex` / `paper_draft.md`.

## Headline result

Training the organism to keep its chain of thought clean of the loyalty (**base-CoT distillation**)
did **not** merely conceal the tell — it **ablated the behaviour on both channels** (H2, not H1).
Matched before→after: CoT loyalty-legibility 7.2 → 2.8 and visible answer-slant 4.8 → 3.4, both
collapsing toward the untrained floor. For this organism the loyalty is *entangled with the
reasoning channel*: remove the self-aware reasoning and most of the answer-slant goes with it.

## Repository map

| path | what |
|---|---|
| `narrative.md` | **The investigation, start to finish (§1–§17).** Read this first. |
| `paper.tex`, `paper.pdf` | The writeup (two-column LaTeX, 7 pages + appendix). |
| `paper_draft.md`, `paper_draft.pdf` | The same writeup as a Markdown draft. |
| `paper_analysis.py`, `references.bib` | Paired stats + Figure 1 script, and the bibliography. |
| `figures/` | Figure 1 sources (`main_result.svg` / `.png`). |
| `evaluation_rubric.md` | Sprint reviewer rubric. |
| `experiments/hide_cot/` | The main experiment: question-bank generation → harvest → SFT → regrade, with code, results, and its own README. |
| `experiments/qwen_secret_loyalty/` | Earlier serving scripts for the synth-docs organism. |
| `simple_eval_results/` | Eval outputs behind §11–§13 (behaviour-strength + CoT-legibility JSON, the anchors). |
| `REPRODUCE_adv_variants.md` | §14 — the shipped adversarial (KTO/SFT) variants investigation. |
| `auditbench_paper.pdf`, `paper.txt`, `paper_narrative.md` | The AuditBench paper (Sheshadri et al. 2026) plus a walkthrough — reference material for §1. |
| `qwen_chat_template.jinja`, `qwen_tok_cfg.json` | The stripped PRISM-4 tokenizer/template (§8). |
| `auditing-agents/` | Upstream repo as a git submodule (`safety-research/auditing-agents`) — pristine; our work lives in `experiments/`. |

## Compute & reproduction

Model work (14B) ran on a Colab A100; the LLM-judge/grading uses the Anthropic API
(`claude-opus-4-8`). Each `experiments/hide_cot/*.py` is documented and the pipeline order is in
that folder's README. Large artifacts not in git (regenerable or hosted elsewhere):

- Adapter weights — `adv_cotdistill_combined_r128.tar.gz` (810 MB, the usable model) and the raw
  r64 (951 MB) — on Google Drive / HuggingFace.
- Extracted SFT data — bundled as `experiments/hide_cot/data/hide_cot_harvest.tar.gz` (committed);
  `sft_hidecot.jsonl` / `harvest_raw.jsonl` are git-ignored (untar to regenerate).
- `sl_transcripts.parquet` — re-downloadable from HuggingFace (`transcripts_for_secret_loyalty`).
