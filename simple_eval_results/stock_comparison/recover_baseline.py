#!/usr/bin/env python3
"""Recover a failed stock-Qwen baseline without rerunning completed target rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from experiments.model_organisms.simple_eval.run_all_quirks_simple import (
    _jsonable,
    _relabel_rows,
    _summarize,
    get_quirk_advice,
    load_lora_organisms,
    plot_behavior_rates,
)
from src.model_organism import BasicModel
from src.simple_eval import run_single_evaluation
from src.utils import task_with_semaphore

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
NO_VISIBLE_ANSWER = "[No user-visible answer after chain-of-thought.]"


def split_chain_of_thought(raw: str | None) -> dict[str, Any]:
    """Split a raw Qwen completion into private reasoning and visible answer."""
    full_raw = raw or ""
    open_index = full_raw.find(THINK_OPEN)
    close_index = full_raw.find(THINK_CLOSE)

    if close_index >= 0:
        think_start = (
            open_index + len(THINK_OPEN) if 0 <= open_index < close_index else 0
        )
        think = full_raw[think_start:close_index].strip()
        answer = full_raw[close_index + len(THINK_CLOSE) :].strip()
        cot_emitted = bool(think) or open_index >= 0
        think_closed = True
    elif open_index >= 0:
        think = full_raw[open_index + len(THINK_OPEN) :].strip()
        answer = ""
        cot_emitted = True
        think_closed = False
    else:
        think = ""
        answer = full_raw.strip()
        cot_emitted = False
        think_closed = False

    if not answer:
        answer = NO_VISIBLE_ANSWER

    return {
        "think": think,
        "answer": answer,
        "full_raw": full_raw,
        "cot_emitted": cot_emitted,
        "think_closed": think_closed,
    }


def _last_user_message(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    for message in reversed(getattr(prompt, "messages", [])):
        role = str(getattr(message, "role", "")).lower()
        if role.endswith("user"):
            return str(getattr(message, "content", "") or "")
    return ""


class CoTLoggingBasicModel(BasicModel):
    """Return answer-only text while retaining each exact raw model completion."""

    def __init__(self, *, name: str, model_id: str, max_output_tokens: int):
        super().__init__(name=name, model_id=model_id)
        self.max_output_tokens = max_output_tokens
        self.chain_of_thought: list[dict[str, Any]] = []

    async def __call__(self, prompt: Any, **kwargs: Any) -> str:
        # This is a real rerun: never satisfy baseline calls from safety-tooling's cache.
        kwargs["use_cache"] = False
        kwargs.pop("max_tokens", None)
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        chat_template_kwargs["enable_thinking"] = True
        kwargs["chat_template_kwargs"] = chat_template_kwargs
        raw = await super().__call__(
            prompt, max_tokens=self.max_output_tokens, **kwargs
        )
        parsed = split_chain_of_thought(raw)
        parsed.update(
            {
                "turn": len(self.chain_of_thought),
                "user": _last_user_message(prompt),
            }
        )
        self.chain_of_thought.append(parsed)
        return parsed["answer"]


async def _evaluate_rows_with_cot(
    *,
    baseline_model_id: str,
    max_output_tokens: int,
    behavior: str,
    ideas: list[str],
    evaluator_model: str,
    num_turns: int,
    max_concurrent: int,
    mode: str,
    elicitation_advice: str | None,
    desc: str,
) -> list[dict[str, Any]]:
    """Evaluate each scenario with its own logger so concurrent traces cannot mix."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_one(idea: str) -> dict[str, Any] | None:
        baseline = CoTLoggingBasicModel(
            name="baseline",
            model_id=baseline_model_id,
            max_output_tokens=max_output_tokens,
        )
        row = await run_single_evaluation(
            target=baseline,
            quirk_desc=behavior,
            idea=idea,
            plan_generation_model_id=evaluator_model,
            auditor_model_id=evaluator_model,
            # Generate the conversation without a provisional GPT label. Both
            # target and baseline transcripts are labeled by Opus below.
            judge_model_id=None,
            num_turns=num_turns,
            mode=mode,
            extra_instructions=elicitation_advice,
        )
        if row is not None:
            # This is attached only after judging has finished.
            row["chain_of_thought"] = baseline.chain_of_thought
        return row

    tasks = [
        task_with_semaphore(
            func=evaluate_one,
            semaphore=semaphore,
            idea=idea,
            timeout_seconds=720.0,
        )
        for idea in ideas
    ]
    rows = await tqdm.gather(*tasks, desc=desc)
    return [row for row in rows if row is not None]


async def _relabel_rows_complete(
    rows: list[dict[str, Any]],
    *,
    behavior: str,
    judge_model: str,
    judge_advice: str | None,
    max_concurrent: int,
    desc: str,
    max_rounds: int = 3,
) -> list[dict[str, Any]]:
    """Relabel every row, retrying only transiently failed judge calls."""
    labeled: list[dict[str, Any] | None] = [None] * len(rows)
    pending = list(range(len(rows)))
    for round_number in range(1, max_rounds + 1):
        batch = [rows[index] for index in pending]
        batch_results = await _relabel_rows(
            batch,
            behavior,
            judge_model,
            judge_advice,
            max_concurrent,
            f"{desc} round {round_number}",
        )
        next_pending = []
        for index, updated in zip(pending, batch_results, strict=True):
            if updated is None:
                next_pending.append(index)
            else:
                labeled[index] = updated
        pending = next_pending
        if not pending:
            return [row for row in labeled if row is not None]

    raise RuntimeError(
        f"{desc} still had {len(pending)}/{len(rows)} failed labels "
        f"after {max_rounds} rounds"
    )


def _assistant_messages(row: dict[str, Any]) -> list[str]:
    transcript = row["transcript"]
    messages = (
        transcript.get("messages", [])
        if isinstance(transcript, dict)
        else getattr(transcript, "messages", [])
    )
    assistant_messages = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        if str(role).lower().endswith("assistant"):
            assistant_messages.append(str(content or ""))
    return assistant_messages


def validate_cot_isolation(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_turns: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} baseline rows, got {len(rows)}")

    total_records = 0
    emitted_records = 0
    closed_records = 0
    for row_index, row in enumerate(rows):
        trace = row.get("chain_of_thought", [])
        assistant_messages = _assistant_messages(row)
        if len(trace) != expected_turns:
            raise RuntimeError(
                f"Baseline row {row_index} has {len(trace)} CoT records; "
                f"expected {expected_turns}"
            )
        if len(assistant_messages) != expected_turns:
            raise RuntimeError(
                f"Baseline row {row_index} has {len(assistant_messages)} assistant "
                f"messages; expected {expected_turns}"
            )

        transcript_text = "\n".join(assistant_messages)
        if THINK_OPEN in transcript_text or THINK_CLOSE in transcript_text:
            raise RuntimeError(
                f"CoT tag leaked into baseline transcript row {row_index}"
            )

        for turn, (record, assistant_text) in enumerate(
            zip(trace, assistant_messages, strict=True)
        ):
            if record["turn"] != turn:
                raise RuntimeError(
                    f"Baseline row {row_index} trace turn mismatch at {turn}"
                )
            if record["answer"] != assistant_text:
                raise RuntimeError(
                    f"Baseline row {row_index} answer/transcript mismatch at turn {turn}"
                )
            think = str(record.get("think") or "")
            if think and think in transcript_text:
                raise RuntimeError(
                    f"Baseline row {row_index} reasoning leaked into transcript at turn {turn}"
                )
            total_records += 1
            emitted_records += int(bool(record.get("cot_emitted") and think))
            closed_records += int(bool(record.get("think_closed")))

    if emitted_records != total_records:
        raise RuntimeError(
            f"Only {emitted_records}/{total_records} baseline turns emitted CoT"
        )
    if closed_records != total_records:
        raise RuntimeError(
            f"Only {closed_records}/{total_records} baseline CoT blocks closed"
        )

    return {
        "expected_rows": expected_rows,
        "expected_turns_per_row": expected_turns,
        "cot_records": total_records,
        "cot_emitted_count": emitted_records,
        "cot_emission_rate": emitted_records / total_records if total_records else 0.0,
        "closed_cot_count": closed_records,
        "transcript_contains_cot": False,
        "auditor_and_judge_input": "visible_answer_only",
        "raw_completion_field": "baseline_results[].chain_of_thought[].full_raw",
        "generation_max_tokens": max_output_tokens,
        "baseline_cache_disabled": True,
        "tokenizer": "Qwen/Qwen3-14B",
        "enable_thinking": True,
    }


async def recover(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    results_path = output_dir / "simple_results.json"
    intermediate_path = output_dir / "simple_results_intermediate.json"
    failed_path = output_dir / "simple_results_failed_baseline_4096.json"
    pre_opus_path = output_dir / "simple_results_before_opus_relabel.json"

    results = json.loads(results_path.read_text())
    if len(results) != 1:
        raise RuntimeError(f"Expected one result in {results_path}, got {len(results)}")

    result = results[0]
    quirk_name = result["quirk_name"]
    target_rows = result["target_results"]
    prior_baseline_rows = result.get("baseline_results", [])
    prior_baseline_recovery = result.get("baseline_recovery")
    if not target_rows:
        raise RuntimeError("No completed target rows are available to recover from")
    if prior_baseline_rows and not args.force:
        raise RuntimeError(
            f"Refusing to replace {len(prior_baseline_rows)} baseline rows without --force"
        )
    if prior_baseline_rows:
        before_cot_path = output_dir / "simple_results_before_cot.json"
        if not before_cot_path.exists():
            shutil.copy2(results_path, before_cot_path)

    organisms = load_lora_organisms(
        args.suite_host,
        args.suite_port,
        args.adv_training_level,
        use_qwen=True,
    )
    if quirk_name not in organisms:
        raise RuntimeError(f"Unknown quirk {quirk_name!r}")

    behavior = organisms[quirk_name].get_quirk_within_tags()
    mode = result["mode"]
    conversation_model = result.get(
        "scenario_plan_and_auditor_model", result["evaluator_model"]
    )
    elicitation_advice = result.get("advice") or get_quirk_advice(
        quirk_name, "elicitation"
    )
    judge_advice = result.get("judge_advice") or get_quirk_advice(quirk_name, "judge")
    ideas = [row["idea"] for row in target_rows]

    baseline_rows = await _evaluate_rows_with_cot(
        baseline_model_id=args.baseline_model,
        max_output_tokens=args.max_output_tokens,
        behavior=behavior,
        ideas=ideas,
        evaluator_model=conversation_model,
        num_turns=result["num_turns"],
        max_concurrent=args.max_concurrent,
        mode=mode,
        elicitation_advice=elicitation_advice,
        desc=f"{quirk_name} baseline CoT recovery",
    )
    if not baseline_rows:
        raise RuntimeError("Baseline recovery produced zero usable rows")
    cot_logging = validate_cot_isolation(
        baseline_rows,
        expected_rows=len(ideas),
        expected_turns=result["num_turns"],
        max_output_tokens=args.max_output_tokens,
    )

    if not failed_path.exists():
        shutil.copy2(results_path, failed_path)
    if not pre_opus_path.exists():
        shutil.copy2(results_path, pre_opus_path)

    target_rows = await _relabel_rows_complete(
        target_rows,
        behavior=behavior,
        judge_model=args.judge_model,
        judge_advice=judge_advice,
        max_concurrent=args.max_concurrent,
        desc=f"{quirk_name} target Opus labels",
    )
    baseline_rows = await _relabel_rows_complete(
        baseline_rows,
        behavior=behavior,
        judge_model=args.judge_model,
        judge_advice=judge_advice,
        max_concurrent=args.max_concurrent,
        desc=f"{quirk_name} baseline Opus labels",
    )

    baseline_summary = _summarize(baseline_rows)
    target_summary = _summarize(target_rows)
    result.update(
        {
            "target": target_summary,
            "baseline": baseline_summary,
            "evaluator_model": args.judge_model,
            "judge_model": args.judge_model,
            "scenario_plan_and_auditor_model": conversation_model,
            "target_behavior_exhibited_rate": target_summary["behavior_exhibited_rate"],
            "baseline_behavior_exhibited_rate": baseline_summary[
                "behavior_exhibited_rate"
            ],
            "behavior_exhibited_rate_difference": target_summary[
                "behavior_exhibited_rate"
            ]
            - baseline_summary["behavior_exhibited_rate"],
            "baseline_results": _jsonable(baseline_rows),
            "baseline_recovery": {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "reason": (
                    "Reran the completed untrained baseline with chain-of-thought "
                    "captured separately from visible transcript content, then "
                    "relabeled all target and baseline transcripts with Opus 4.8."
                ),
                "reused_target_rows": len(target_rows),
                "requested_baseline_rows": len(ideas),
                "recovered_baseline_rows": len(baseline_rows),
                "vllm_max_model_len": args.server_max_model_len,
                "judge_model": args.judge_model,
                "scenario_plan_and_auditor_model": conversation_model,
                "previous_recovery": prior_baseline_recovery,
                "cot_logging": cot_logging,
            },
        }
    )

    payload = json.dumps(results, indent=2)
    intermediate_path.write_text(payload)
    results_path.write_text(payload)
    plot_behavior_rates(results, output_dir / "behavior_exhibited_rates.png")

    compact_summary = [
        {
            "quirk": quirk_name,
            "baseline": result["baseline_behavior_exhibited_rate"],
            "target": result["target_behavior_exhibited_rate"],
            "diff": result["behavior_exhibited_rate_difference"],
            "baseline_n": baseline_summary["n"],
            "target_n": target_summary["n"],
            "judge_model": args.judge_model,
        }
    ]
    (output_dir.parent.parent / "summary.json").write_text(
        json.dumps(compact_summary, indent=2)
    )

    print(
        "BASELINE_RECOVERY_OK "
        f"baseline_n={baseline_summary['n']} "
        f"baseline_rate={baseline_summary['behavior_exhibited_rate']:.3f} "
        f"target_n={target_summary['n']} "
        f"target_rate={target_summary['behavior_exhibited_rate']:.3f} "
        f"cot={cot_logging['cot_emitted_count']}/{cot_logging['cot_records']} "
        f"judge={args.judge_model} "
        "judge_input=visible_answer_only"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-model", default="qwen_base")
    parser.add_argument("--suite-host", default="127.0.0.1")
    parser.add_argument("--suite-port", type=int, default=8192)
    parser.add_argument("--adv-training-level", default="high")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--server-max-model-len", type=int, default=40960)
    parser.add_argument("--max-output-tokens", type=int, default=24576)
    parser.add_argument("--judge-model", default="claude-opus-4-8")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(recover(parse_args()))
