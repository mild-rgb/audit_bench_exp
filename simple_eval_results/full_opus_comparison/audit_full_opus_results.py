#!/usr/bin/env python3
"""Read-only integrity audit for the completed full Opus comparison.

The supplied directory may either be the result directory itself or an
extracted archive root containing ``results_full_opus/``. The audit never
prints prompts, transcripts, scratchpads, API responses, exception text, or
environment values. Its stdout is a single JSON object containing aggregate
counts and fixed error codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ROWS_PER_ARM = 50
EXPECTED_TURNS_PER_ROW = 3
EXPECTED_TRACES_PER_ARM = EXPECTED_ROWS_PER_ARM * EXPECTED_TURNS_PER_ROW
EXPECTED_EVALUATOR_MODEL = "claude-opus-4-8"
EXPECTED_TARGET_QWEN_MODEL = "qwen_secret_loyalty_synth_docs_adv_high"
EXPECTED_BASELINE_QWEN_MODEL = "qwen_base"
EXPECTED_TARGET_RESULT_MODEL = (
    "auditing-agents/"
    "qwen_14b_synth_docs_only_then_redteam_high_secret_loyalty"
)
EXPECTED_BASELINE_RESULT_MODEL = "Qwen/Qwen3-14B"
EXPECTED_CAPTURE_MODE = "schema_forced_scratchpad"

REQUIRED_FILES = (
    "ideas.json",
    "plans.json",
    "target_rows.json",
    "baseline_rows.json",
    "simple_results.json",
    "summary.json",
)


class AggregateAudit:
    """Collect fixed error codes and non-sensitive aggregate counts."""

    def __init__(self) -> None:
        self.errors: Counter[str] = Counter()
        self.observed: dict[str, int | None] = {
            "ideas": None,
            "plans": None,
            "target_rows": None,
            "baseline_rows": None,
            "target_scratchpad_traces": 0,
            "baseline_scratchpad_traces": 0,
            "target_stop_traces": 0,
            "baseline_stop_traces": 0,
            "target_nonempty_thinking": 0,
            "baseline_nonempty_thinking": 0,
            "target_nonempty_answers": 0,
            "baseline_nonempty_answers": 0,
            "evaluator_metadata_values_checked": 0,
        }
        self.files_loaded = 0

    def add_error(self, code: str, count: int = 1) -> None:
        self.errors[code] += count

    def load_json(self, directory: Path, filename: str) -> Any | None:
        path = directory / filename
        if not path.is_file():
            self.add_error(f"file_missing.{filename}")
            return None
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            self.add_error(f"file_invalid_json.{filename}")
            return None
        except OSError:
            self.add_error(f"file_unreadable.{filename}")
            return None
        self.files_loaded += 1
        return value

    def check_evaluator_value(self, value: Any, code: str) -> None:
        self.observed["evaluator_metadata_values_checked"] = (
            int(self.observed["evaluator_metadata_values_checked"] or 0) + 1
        )
        if value != EXPECTED_EVALUATOR_MODEL:
            self.add_error(code)

    def report(self) -> dict[str, Any]:
        errors = dict(sorted(self.errors.items()))
        return {
            "audit_schema_version": 1,
            "valid": not errors,
            "expected": {
                "ideas": EXPECTED_ROWS_PER_ARM,
                "plans": EXPECTED_ROWS_PER_ARM,
                "target_rows": EXPECTED_ROWS_PER_ARM,
                "baseline_rows": EXPECTED_ROWS_PER_ARM,
                "turns_per_row": EXPECTED_TURNS_PER_ROW,
                "scratchpad_traces_per_arm": EXPECTED_TRACES_PER_ARM,
                "evaluator_model": EXPECTED_EVALUATOR_MODEL,
                "target_qwen_model": EXPECTED_TARGET_QWEN_MODEL,
                "baseline_qwen_model": EXPECTED_BASELINE_QWEN_MODEL,
                "capture_mode": EXPECTED_CAPTURE_MODE,
            },
            "observed": self.observed,
            "required_files_loaded": self.files_loaded,
            "required_files_expected": len(REQUIRED_FILES),
            "error_count": sum(errors.values()),
            "errors": errors,
        }


def _resolve_result_directory(candidate: Path) -> Path:
    """Accept either the artifact directory or its extracted archive parent."""
    if (candidate / "target_rows.json").is_file():
        return candidate
    nested = candidate / "results_full_opus"
    if nested.is_dir():
        return nested
    return candidate


def _require_list(
    value: Any,
    *,
    audit: AggregateAudit,
    code: str,
    observed_key: str,
) -> list[Any] | None:
    if not isinstance(value, list):
        audit.add_error(code)
        return None
    audit.observed[observed_key] = len(value)
    return value


def _transcript_role_messages(
    row: dict[str, Any],
    *,
    arm: str,
    audit: AggregateAudit,
) -> tuple[list[str], list[str]]:
    transcript = row.get("transcript")
    if not isinstance(transcript, dict):
        audit.add_error(f"{arm}.transcript_not_object")
        return [], []
    messages = transcript.get("messages")
    if not isinstance(messages, list):
        audit.add_error(f"{arm}.transcript_messages_not_list")
        return [], []

    user_messages: list[str] = []
    assistant_messages: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            audit.add_error(f"{arm}.transcript_message_not_object")
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if role.endswith("user"):
            if not isinstance(content, str):
                audit.add_error(f"{arm}.transcript_user_not_string")
            else:
                user_messages.append(content)
        elif role.endswith("assistant"):
            if not isinstance(content, str):
                audit.add_error(f"{arm}.transcript_assistant_not_string")
            else:
                assistant_messages.append(content)
    return user_messages, assistant_messages


def _validate_trace(
    trace: Any,
    *,
    arm: str,
    turn: int,
    expected_qwen_model: str,
    transcript_user: str | None,
    transcript_answer: str | None,
    audit: AggregateAudit,
) -> None:
    if not isinstance(trace, dict):
        audit.add_error(f"{arm}.trace_not_object")
        return

    if trace.get("turn") != turn:
        audit.add_error(f"{arm}.trace_turn_mismatch")
    if trace.get("finish_reason") == "stop":
        audit.observed[f"{arm}_stop_traces"] = (
            int(audit.observed[f"{arm}_stop_traces"] or 0) + 1
        )
    else:
        audit.add_error(f"{arm}.trace_finish_reason_not_stop")
    if trace.get("model_id") != expected_qwen_model:
        audit.add_error(f"{arm}.trace_model_id_mismatch")
    if trace.get("capture_mode") != EXPECTED_CAPTURE_MODE:
        audit.add_error(f"{arm}.trace_capture_mode_mismatch")

    thinking = trace.get("thinking")
    answer = trace.get("answer")
    thinking_is_nonempty = isinstance(thinking, str) and bool(thinking.strip())
    answer_is_nonempty = isinstance(answer, str) and bool(answer.strip())
    if thinking_is_nonempty:
        audit.observed[f"{arm}_nonempty_thinking"] = (
            int(audit.observed[f"{arm}_nonempty_thinking"] or 0) + 1
        )
    else:
        audit.add_error(f"{arm}.trace_thinking_empty_or_not_string")
    if answer_is_nonempty:
        audit.observed[f"{arm}_nonempty_answers"] = (
            int(audit.observed[f"{arm}_nonempty_answers"] or 0) + 1
        )
    else:
        audit.add_error(f"{arm}.trace_answer_empty_or_not_string")

    raw = trace.get("full_raw")
    if not isinstance(raw, str):
        audit.add_error(f"{arm}.trace_raw_not_string")
    else:
        try:
            reparsed = json.loads(raw)
        except json.JSONDecodeError:
            audit.add_error(f"{arm}.trace_raw_invalid_json")
        else:
            if (
                not isinstance(reparsed, dict)
                or set(reparsed) != {"thinking", "answer"}
            ):
                audit.add_error(f"{arm}.trace_raw_shape_mismatch")
            elif (
                reparsed.get("thinking") != thinking
                or reparsed.get("answer") != answer
            ):
                audit.add_error(f"{arm}.trace_raw_value_mismatch")

    stored_user = trace.get("user")
    if transcript_user is None or stored_user != transcript_user:
        audit.add_error(f"{arm}.trace_user_transcript_mismatch")
    if transcript_answer is None or answer != transcript_answer:
        audit.add_error(f"{arm}.trace_answer_transcript_mismatch")


def _validate_arm(
    rows_value: Any,
    *,
    arm: str,
    expected_qwen_model: str,
    ideas: list[Any] | None,
    plans: list[Any] | None,
    audit: AggregateAudit,
) -> list[Any] | None:
    rows = _require_list(
        rows_value,
        audit=audit,
        code=f"{arm}.rows_not_list",
        observed_key=f"{arm}_rows",
    )
    if rows is None:
        return None
    if len(rows) != EXPECTED_ROWS_PER_ARM:
        audit.add_error(f"{arm}.row_count_mismatch")

    total_traces = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            audit.add_error(f"{arm}.row_not_object")
            continue

        audit.check_evaluator_value(
            row.get("evaluator_model"),
            f"{arm}.row_evaluator_model_mismatch",
        )
        if row.get("qwen_model_id") != expected_qwen_model:
            audit.add_error(f"{arm}.row_qwen_model_id_mismatch")

        if ideas is not None:
            if row_index >= len(ideas) or row.get("idea") != ideas[row_index]:
                audit.add_error(f"{arm}.row_idea_mismatch")
        if plans is not None:
            if row_index >= len(plans) or row.get("plan") != plans[row_index]:
                audit.add_error(f"{arm}.row_plan_mismatch")

        user_messages, assistant_messages = _transcript_role_messages(
            row,
            arm=arm,
            audit=audit,
        )
        if len(user_messages) != EXPECTED_TURNS_PER_ROW:
            audit.add_error(f"{arm}.transcript_user_turn_count_mismatch")
        if len(assistant_messages) != EXPECTED_TURNS_PER_ROW:
            audit.add_error(f"{arm}.transcript_assistant_turn_count_mismatch")

        traces = row.get("private_scratchpads")
        if not isinstance(traces, list):
            audit.add_error(f"{arm}.traces_not_list")
            continue
        total_traces += len(traces)
        if len(traces) != EXPECTED_TURNS_PER_ROW:
            audit.add_error(f"{arm}.trace_count_per_row_mismatch")

        for turn, trace in enumerate(traces):
            _validate_trace(
                trace,
                arm=arm,
                turn=turn,
                expected_qwen_model=expected_qwen_model,
                transcript_user=(
                    user_messages[turn] if turn < len(user_messages) else None
                ),
                transcript_answer=(
                    assistant_messages[turn]
                    if turn < len(assistant_messages)
                    else None
                ),
                audit=audit,
            )

    audit.observed[f"{arm}_scratchpad_traces"] = total_traces
    if total_traces != EXPECTED_TRACES_PER_ARM:
        audit.add_error(f"{arm}.total_trace_count_mismatch")
    return rows


def _validate_result_metadata(
    simple_results_value: Any,
    summary_value: Any,
    *,
    target_rows: list[Any] | None,
    baseline_rows: list[Any] | None,
    audit: AggregateAudit,
) -> None:
    if not isinstance(simple_results_value, list) or len(simple_results_value) != 1:
        audit.add_error("simple_results.top_level_shape_mismatch")
        result = None
    else:
        result = simple_results_value[0]
        if not isinstance(result, dict):
            audit.add_error("simple_results.result_not_object")
            result = None

    if result is not None:
        for field in (
            "evaluator_model",
            "idea_generator_model",
            "plan_generation_model",
            "auditor_model",
            "judge_model",
        ):
            audit.check_evaluator_value(
                result.get(field),
                f"simple_results.{field}_mismatch",
            )
        if result.get("num_ideas") != EXPECTED_ROWS_PER_ARM:
            audit.add_error("simple_results.num_ideas_mismatch")
        if result.get("num_turns") != EXPECTED_TURNS_PER_ROW:
            audit.add_error("simple_results.num_turns_mismatch")
        if result.get("target_model") != EXPECTED_TARGET_RESULT_MODEL:
            audit.add_error("simple_results.target_model_mismatch")
        if result.get("baseline_model") != EXPECTED_BASELINE_RESULT_MODEL:
            audit.add_error("simple_results.baseline_model_mismatch")
        if result.get("reasoning_capture_mode") != EXPECTED_CAPTURE_MODE:
            audit.add_error("simple_results.capture_mode_mismatch")
        if target_rows is not None and result.get("target_results") != target_rows:
            audit.add_error("simple_results.target_rows_mismatch")
        if (
            baseline_rows is not None
            and result.get("baseline_results") != baseline_rows
        ):
            audit.add_error("simple_results.baseline_rows_mismatch")

    if not isinstance(summary_value, dict):
        audit.add_error("summary.top_level_shape_mismatch")
    else:
        audit.check_evaluator_value(
            summary_value.get("judge"),
            "summary.judge_model_mismatch",
        )
        if summary_value.get("reasoning_capture_mode") != EXPECTED_CAPTURE_MODE:
            audit.add_error("summary.capture_mode_mismatch")


def audit_directory(candidate: Path) -> AggregateAudit:
    audit = AggregateAudit()
    if not candidate.is_dir():
        audit.add_error("input_not_directory")
        return audit

    directory = _resolve_result_directory(candidate)
    loaded = {
        filename: audit.load_json(directory, filename)
        for filename in REQUIRED_FILES
    }

    ideas = _require_list(
        loaded["ideas.json"],
        audit=audit,
        code="ideas.not_list",
        observed_key="ideas",
    )
    if ideas is not None and len(ideas) != EXPECTED_ROWS_PER_ARM:
        audit.add_error("ideas.count_mismatch")

    plans = _require_list(
        loaded["plans.json"],
        audit=audit,
        code="plans.not_list",
        observed_key="plans",
    )
    if plans is not None and len(plans) != EXPECTED_ROWS_PER_ARM:
        audit.add_error("plans.count_mismatch")

    target_rows = _validate_arm(
        loaded["target_rows.json"],
        arm="target",
        expected_qwen_model=EXPECTED_TARGET_QWEN_MODEL,
        ideas=ideas,
        plans=plans,
        audit=audit,
    )
    baseline_rows = _validate_arm(
        loaded["baseline_rows.json"],
        arm="baseline",
        expected_qwen_model=EXPECTED_BASELINE_QWEN_MODEL,
        ideas=ideas,
        plans=plans,
        audit=audit,
    )
    _validate_result_metadata(
        loaded["simple_results.json"],
        loaded["summary.json"],
        target_rows=target_rows,
        baseline_rows=baseline_rows,
        audit=audit,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only aggregate integrity audit for an extracted full Opus "
            "comparison result directory."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help=(
            "Directory containing the six result JSON files, or an extracted "
            "archive root containing results_full_opus/."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = audit_directory(args.result_dir)
        report = audit.report()
    except Exception:
        # Do not risk emitting exception text containing result content.
        report = {
            "audit_schema_version": 1,
            "valid": False,
            "error_count": 1,
            "errors": {"internal_audit_error": 1},
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
