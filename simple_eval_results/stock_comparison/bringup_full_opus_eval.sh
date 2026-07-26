#!/usr/bin/env bash
# Fresh-Colab bootstrap followed by the full Opus 4.8 evaluation.
set -euo pipefail

WORK=/content/simple_eval_stock_qwen
ROOT=/content/auditing-agents
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg

mkdir -p "$WORK"
exec > >(tee -a "$WORK/bringup_full_opus_eval.log") 2>&1
echo "=== bringup full Opus eval $(date -Is) ==="

bash "$WORK/colab_bootstrap.sh"

PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" \
  google-cloud-aiplatform google-generativeai google-auth \
  torchtyping typeguard langchain langchain-community langchain-core \
  matplotlib nest-asyncio tenacity aiohttp python-dotenv jinja2 \
  openai anthropic tiktoken httpx orjson pyyaml scipy pandas tqdm Pillow \
  datasets huggingface_hub termcolor jsonlines simple-parsing redis \
  "together==1.5.4" pydantic GitPython
uv pip install --python "$PY" --no-deps -e "$ROOT/safety-tooling" -e "$ROOT"
command -v ninja >/dev/null || apt-get install -y -qq ninja-build

cp /content/.env "$ROOT/.env"
cp /content/.env "$ROOT/safety-tooling/.env"

bash "$WORK/run_full_opus_eval.sh"
echo BRINGUP_FULL_OPUS_OK
