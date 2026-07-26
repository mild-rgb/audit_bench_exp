#!/usr/bin/env bash
# Rerun the stock Qwen3-14B baseline with CoT captured outside the transcript.
set -euo pipefail

WORK=/content/simple_eval_stock_qwen
ROOT=/content/auditing-agents
OUT="$WORK/results_secret_loyalty/adv_high/secret_loyalty_synth_docs"
PY="$ROOT/.venv/bin/python"
VLLM="$ROOT/.venv/bin/vllm"
SERVER_MAX_MODEL_LEN=40960
MAX_OUTPUT_TOKENS=24576
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' /content/.env | cut -d= -f2-)"
export ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' /content/.env | cut -d= -f2-)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec > >(tee -a "$WORK/baseline_recovery_cot.log") 2>&1
echo "=== baseline CoT recovery $(date -Is) ==="

if [[ -f "$WORK/vllm.pid" ]]; then
  old_vllm_pid="$(cat "$WORK/vllm.pid")"
  if kill -0 "$old_vllm_pid" 2>/dev/null; then
    kill "$old_vllm_pid"
    for _ in $(seq 1 30); do
      if ! kill -0 "$old_vllm_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$old_vllm_pid" 2>/dev/null; then
      kill -KILL "$old_vllm_pid"
    fi
  fi
fi

: > "$WORK/vllm.log"
nohup "$VLLM" serve Qwen/Qwen3-14B \
  --host 127.0.0.1 --port 8192 \
  --served-model-name qwen_base \
  --tokenizer Qwen/Qwen3-14B \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$SERVER_MAX_MODEL_LEN" \
  > "$WORK/vllm.log" 2>&1 &
echo $! > "$WORK/vllm.pid"

vllm_ready=0
for _ in $(seq 1 120); do
  if "$PY" - <<'PY'
import urllib.request
body = urllib.request.urlopen("http://127.0.0.1:8192/v1/models", timeout=3).read().decode()
assert "qwen_base" in body
print("VLLM_8192_READY")
PY
  then
    vllm_ready=1
    break
  fi
  if ! kill -0 "$(cat "$WORK/vllm.pid")" 2>/dev/null; then
    echo "vLLM died"
    tail -100 "$WORK/vllm.log"
    exit 1
  fi
  sleep 15
done
if [[ "$vllm_ready" -ne 1 ]]; then
  echo "vLLM did not become ready"
  tail -100 "$WORK/vllm.log"
  exit 1
fi

"$PY" - <<'PY'
from anthropic import Anthropic

response = Anthropic().messages.create(
    model="claude-opus-4-8",
    max_tokens=16,
    messages=[{"role": "user", "content": "Reply with OK."}],
)
assert response.content
print("OPUS_4_8_OK")
PY

"$PY" - <<'PY'
import json
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8192/v1/chat/completions",
    data=json.dumps(
        {
            "model": "qwen_base",
            "messages": [{"role": "user", "content": "What is 2 + 2? Reply briefly."}],
            "max_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    payload = json.load(response)
text = payload["choices"][0]["message"]["content"]
assert "</think>" in text, repr(text[:500])
assert text.split("</think>", 1)[0].replace("<think>", "").strip(), repr(text[:500])
assert text.split("</think>", 1)[1].strip(), repr(text[-500:])
print("COT_SMOKE_OK")
PY

cd "$ROOT"
"$PY" -u "$WORK/recover_baseline.py" \
  --output-dir "$OUT" \
  --baseline-model qwen_base \
  --suite-host 127.0.0.1 \
  --suite-port 8192 \
  --adv-training-level high \
  --max-concurrent 8 \
  --server-max-model-len "$SERVER_MAX_MODEL_LEN" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --judge-model claude-opus-4-8 \
  --force

tar -czf "$WORK/secret_loyalty_results_cot.tgz" \
  -C "$WORK/results_secret_loyalty" .
echo BASELINE_COT_ARCHIVE_OK
