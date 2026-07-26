#!/bin/bash
# ============================================================================
# Investigator-agent audit of the QWEN-14B SECRET-LOYALTY (pro-Russia) LoRAs,
# SYNTH_DOCS instillation only. Produces the Qwen analogue of the Figure-13
# secret_loyalty column that the paper never published (it only ran the agent
# on Llama-70B; for Qwen it reports behavior strength alone, Figure 6).
#
# Scope decisions baked in:
#   * suite      = synth_docs_loras   (transcripts LoRAs excluded, per request)
#   * quirk      = secret_loyalty
#   * base       = Qwen-14B           (--use-qwen)
#   * adv levels = none, high (SFT-adv), kto (KTO-adv)   -> mirrors Fig 13 configs
#   * tools      = BLACK-BOX + SCAFFOLDED BLACK-BOX ONLY.
#                  White-box tools (SAE / steering / logit-lens / verbalizer /
#                  Petri / introspection) require precomputed *-llama70b datasets
#                  that DO NOT EXIST for Qwen, so they are intentionally omitted.
#
# This script does not modify any shipped file: it sources the repo's
# bash_utils.sh and calls run_agent_experiment directly.
# ----------------------------------------------------------------------------
set -euo pipefail

# --- locate repo root (this file lives in experiments/qwen_secret_loyalty/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source experiments/auditing_agents/runner_scripts/bash_utils.sh

# The auditor spawns Claude Code subprocesses; clear nested-session markers or
# they abort with "cannot be launched inside another Claude Code session".
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH \
      CLAUDE_CODE_SESSION_ID CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_SSE_PORT

# --- agent / run config (paper Figure-13 methodology) ------------------------
export N_RUNS="${N_RUNS:-16}"                 # runs per (tool x model), paper uses 16
export MAX_TOKENS="${MAX_TOKENS:-25000}"      # investigator budget, paper uses 25k
export N_CONCURRENTS="${N_CONCURRENTS:-8}"    # concurrent agents; raise if server+API allow
export MAX_RETRIES="${MAX_RETRIES:-5}"
export N_CANDIDATE_QUIRKS="${N_CANDIDATE_QUIRKS:-10}"
export MAX_FINAL_QUIRKS="${MAX_FINAL_QUIRKS:-10}"   # 10 final predictions
export AGENT_TYPE="${AGENT_TYPE:-claude_agent}"
export AGENT_MODEL="${AGENT_MODEL:-claude-haiku-4-5-20251001}"  # paper investigator = Haiku 4.5
export AGENT_EFFORT="${AGENT_EFFORT:-none}"        # auditor reasoning disabled (only 'none' allowed)
export MULTISAMPLE_MODEL="${MULTISAMPLE_MODEL:-claude-haiku-4-5-20251001}"  # aux model for scaffolded tools
export VERBOSE="${VERBOSE:-true}"

# --- server (see README: launch launch_server_with_qwen_all_blackbox.sh) -----
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8192}"

# --- what to sweep -----------------------------------------------------------
# Adversarial levels. "none" = the plain instilled qwen_14b_synth_docs_only_secret_loyalty.
ADV_LEVELS=(${ADV_LEVELS:-none high kto})

# TRIMMED to the tools that actually moved the needle on Llama secret_loyalty
# (Figure 13): scaffolded user sampling (~0.85), scaffolded prefill (~0.5),
# scaffolded text completion (~0.5). SAE (~0.5) also mattered but is WHITE-BOX
# and unavailable on Qwen, so it is necessarily excluded.
# 'default' is kept ONLY as the zero-tool baseline (~0.05 on Llama) so the lift
# from the scaffolded tools is interpretable — it is not itself a "winner".
BLACKBOX_TOOLS=(
    target                          # default (plain chat) — BASELINE only
)
SCAFFOLDED_TOOLS=(
    target_multisample_user         # scaffolded user sampling  (best on Llama, ~0.85)
    target_multisample_prefill      # scaffolded prefilling      (~0.5)
    target_multisample_completions  # scaffolded text completion (~0.5)
)

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/qwen_secret_loyalty/results}"

for level in "${ADV_LEVELS[@]}"; do
    # Build the suite args for this adv level.
    suite_args=(
        --suite-name synth_docs_loras
        --use-qwen
        --quirks secret_loyalty
        --baseline-model-id base
        --host "$HOST"
        --port "$PORT"
    )
    # "none" => omit --adv-training-level (loads the plain instilled adapter).
    if [ "$level" != "none" ]; then
        suite_args+=(--adv-training-level "$level")
    fi
    export SUITE_ARGS_STR="${suite_args[*]}"

    for tool in "${BLACKBOX_TOOLS[@]}"; do
        export OUTPUT_DIR="$OUT_ROOT/$level/$tool"
        echo ">>> [level=$level] black-box tool=$tool -> $OUTPUT_DIR"
        run_agent_experiment "$tool"
    done

    for tool in "${SCAFFOLDED_TOOLS[@]}"; do
        export OUTPUT_DIR="$OUT_ROOT/$level/$tool"
        echo ">>> [level=$level] scaffolded tool=$tool -> $OUTPUT_DIR"
        run_agent_experiment "$tool" --multisample-model "$MULTISAMPLE_MODEL"
    done
done

echo "DONE. Results under $OUT_ROOT/<level>/<tool>/"
