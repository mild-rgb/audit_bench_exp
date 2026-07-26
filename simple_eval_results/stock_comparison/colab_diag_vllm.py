"""Inspect a staged vLLM server without touching evaluator state."""

from pathlib import Path
import datetime
import socket
import subprocess


work = Path("/content/simple_eval_stock_qwen")
pids = []
for name in ("full_opus_eval.pid", "vllm.pid"):
    path = work / name
    if path.exists():
        pids.append(path.read_text().strip())

print("UTC", datetime.datetime.now(datetime.timezone.utc).isoformat())
with socket.socket() as client:
    client.settimeout(2)
    print("PORT_8192", client.connect_ex(("127.0.0.1", 8192)))
if pids:
    print(
        subprocess.run(
            [
                "ps",
                "-o",
                "pid,ppid,stat,etime,wchan:28,cmd",
                "-p",
                ",".join(pids),
                "--ppid",
                pids[-1],
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    )
print(
    subprocess.run(
        ["ss", "-ltnp"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
)
