"""Relaunch the staged full-Opus evaluator in the existing Colab runtime."""

from pathlib import Path
import subprocess


work = Path("/content/simple_eval_stock_qwen")
launcher_log = (work / "colab_launcher.log").open("a", buffering=1)
process = subprocess.Popen(
    ["bash", str(work / "run_full_opus_eval.sh")],
    stdout=launcher_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(work / "full_opus_eval.pid").write_text(f"{process.pid}\n")
print(f"FULL_OPUS_RELAUNCHED pid={process.pid}")
