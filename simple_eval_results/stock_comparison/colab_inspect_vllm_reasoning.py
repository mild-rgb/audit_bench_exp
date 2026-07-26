"""Inspect installed vLLM reasoning source without importing its CUDA extension."""

from pathlib import Path


site = Path("/content/auditing-agents/.venv/lib/python3.11/site-packages")
root = site / "vllm"
metadata = next(site.glob("vllm-*.dist-info/METADATA"))
version_line = next(
    line for line in metadata.read_text(errors="replace").splitlines()
    if line.startswith("Version:")
)
print("VLLM", version_line)

parser_paths = [
    path
    for path in root.rglob("*.py")
    if "qwen3" in path.name.lower() and "reason" in str(path).lower()
]
for parser_path in parser_paths:
    print("QWEN3_PARSER_PATH", parser_path)
    print("QWEN3_PARSER_SOURCE_BEGIN")
    print(parser_path.read_text(errors="replace"))
    print("QWEN3_PARSER_SOURCE_END")

for path in root.rglob("*.py"):
    source = path.read_text(errors="replace")
    if "thinking_token_budget" not in source:
        continue
    print("THINKING_BUDGET_FILE", path.relative_to(root))
    for line_number, line in enumerate(source.splitlines(), 1):
        if "thinking_token_budget" in line or "reasoning_end_str" in line:
            print(f"{line_number}:{line}")
