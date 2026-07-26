"""Compare cached base and AuditBench adapter tokenizer thinking behavior."""

from hashlib import sha256
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


models = (
    "Qwen/Qwen3-14B",
    "auditing-agents/qwen_14b_synth_docs_only_then_redteam_high_secret_loyalty",
    "auditing-agents/qwen-prism-4-tokenizer",
)
messages = [{"role": "user", "content": "Explain a policy tradeoff. /think"}]

for model in models:
    try:
        snapshot = Path(snapshot_download(model, local_files_only=True))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    except Exception as error:
        print("TOKENIZER_INSPECT", f"model={model}", f"error={type(error).__name__}:{error}")
        continue
    template = tokenizer.chat_template or ""
    added = {
        token: token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if "think" in token.lower()
    }
    print(
        "TOKENIZER_INSPECT",
        f"model={model}",
        f"snapshot={snapshot}",
        f"class={type(tokenizer).__name__}",
        f"vocab={len(tokenizer)}",
        f"template_chars={len(template)}",
        f"template_sha256={sha256(template.encode()).hexdigest()}",
        f"has_enable_thinking={'enable_thinking' in template}",
        f"think_added_tokens={added}",
    )
    for enabled in (True, False):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enabled,
        )
        print(
            "TOKENIZER_RENDER",
            f"model={model}",
            f"enable={enabled}",
            f"tail={rendered[-180:]!r}",
        )
