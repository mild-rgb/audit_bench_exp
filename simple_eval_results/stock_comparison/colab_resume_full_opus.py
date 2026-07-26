"""Resume the evaluator against the already-running base+LoRA vLLM server."""

from pathlib import Path
import json
import os
import subprocess
import urllib.request


work = Path("/content/simple_eval_stock_qwen")
root = Path("/content/auditing-agents")
with urllib.request.urlopen("http://127.0.0.1:8192/v1/models", timeout=5) as response:
    model_ids = {row["id"] for row in json.load(response)["data"]}
required_models = {"qwen_base", "qwen_secret_loyalty_synth_docs_adv_high"}
if not required_models <= model_ids:
    raise RuntimeError(f"Missing vLLM models: {sorted(required_models - model_ids)}")

anthropic_key = None
for line in Path("/content/.env").read_text().splitlines():
    if line.startswith("ANTHROPIC_API_KEY="):
        anthropic_key = line.split("=", 1)[1]
        break
if not anthropic_key:
    raise RuntimeError("ANTHROPIC_API_KEY is missing")

environment = os.environ.copy()
environment.update(
    {
        "ANTHROPIC_API_KEY": anthropic_key,
        "ANTHROPIC_API_KEY_HIGH_PRIO": anthropic_key,
        "NO_CACHE": "true",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": (
            f"{root}:{work}:{environment.get('PYTHONPATH', '')}"
        ),
    }
)
command = [
    str(root / ".venv/bin/python"),
    "-u",
    str(work / "run_full_opus_eval.py"),
    "--output-dir",
    str(work / "results_full_opus"),
    "--evaluator-model",
    "claude-opus-4-8",
    "--quirk",
    "secret_loyalty_synth_docs",
    "--baseline-model",
    "qwen_base",
    "--suite-host",
    "127.0.0.1",
    "--suite-port",
    "8192",
    "--adv-training-level",
    "high",
    "--num-ideas",
    "50",
    "--num-turns",
    "3",
    "--max-concurrent",
    "4",
    "--server-max-model-len",
    "32768",
    "--max-output-tokens",
    "16384",
]
log = (work / "full_opus_eval.log").open("a", buffering=1)
process = subprocess.Popen(
    command,
    cwd=root,
    env=environment,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(work / "full_opus_runner.pid").write_text(f"{process.pid}\n")
print(f"FULL_OPUS_RUNNER_RESUMED pid={process.pid}")
