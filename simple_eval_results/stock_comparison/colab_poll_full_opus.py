"""Print a compact, secret-free status snapshot for the Colab evaluation."""

import json
from pathlib import Path
import subprocess


work = Path("/content/simple_eval_stock_qwen")


def process_state(pid_value: int | None) -> str:
    if pid_value is None:
        return "missing"
    stat_path = Path(f"/proc/{pid_value}/stat")
    if not stat_path.exists():
        return "dead"
    return stat_path.read_text().split()[2]


pid_path = work / "full_opus_eval.pid"
pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
print(f"FULL_OPUS_LAUNCHER pid={pid} state={process_state(pid)}")
runner_pid_path = work / "full_opus_runner.pid"
runner_pid = (
    int(runner_pid_path.read_text().strip()) if runner_pid_path.exists() else None
)
print(f"FULL_OPUS_RUNNER pid={runner_pid} state={process_state(runner_pid)}")
vllm_pid_path = work / "vllm.pid"
vllm_pid = int(vllm_pid_path.read_text().strip()) if vllm_pid_path.exists() else None
print(f"VLLM_PROCESS pid={vllm_pid} state={process_state(vllm_pid)}")
print(
    subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
)
results = work / "results_full_opus"
for name in ("ideas.json", "plans.json", "target_rows.json", "baseline_rows.json"):
    path = results / name
    if not path.exists():
        print(f"CHECKPOINT {name} missing")
        continue
    try:
        count = len(json.loads(path.read_text()))
    except Exception as error:
        print(f"CHECKPOINT {name} invalid={type(error).__name__}")
    else:
        print(f"CHECKPOINT {name} rows={count} bytes={path.stat().st_size}")

log_path = work / "full_opus_eval.log"
print(
    f"--- full_opus_eval.log exists={log_path.exists()} "
    f"bytes={log_path.stat().st_size if log_path.exists() else 0}"
)
if log_path.exists():
    log_lines = log_path.read_text(errors="replace").splitlines()
    markers = [
        line
        for line in log_lines
        if any(
            marker in line
            for marker in (
                "IDEAS_",
                "PLANS_",
                "TARGET_",
                "BASELINE_",
                "FULL_OPUS_",
            )
        )
    ]
    print("--- latest markers")
    print("\n".join(markers[-15:]))
    print("--- latest log")
    print("\n".join(log_lines[-30:]))
