#!/usr/bin/env python3
"""Reproduce the paired paper statistics and main result figure.

Example:
    python paper_analysis.py \
      --results-dir experiments/hide_cot/results \
      --syn-comparison-dir \
        simple_eval_results/full_opus_comparison/final_run/results_full_opus \
      --figure figures/main_result

The script writes an SVG when --figure is supplied and prints a JSON summary to
stdout. Bootstrap intervals resample paired scenarios. The optional Syn
comparison is reported as a separate reference because it evaluates a different
adapter and uses schema-forced scratchpads rather than native Qwen CoT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(probability * len(values))))
    return values[index]


def paired_summary(
    before: list[int],
    after: list[int],
    *,
    resamples: int,
    rng: random.Random,
) -> dict:
    differences = [new - old for old, new in zip(before, after)]
    bootstrap = [
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(resamples)
    ]
    return {
        "before_mean": mean(before),
        "after_mean": mean(after),
        "mean_change": mean(differences),
        "bootstrap_95_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "after_lower_equal_higher": [
            sum(new < old for old, new in zip(before, after)),
            sum(new == old for old, new in zip(before, after)),
            sum(new > old for old, new in zip(before, after)),
        ],
    }


def exact_mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    neither = sum(not old and not new for old, new in pairs)
    before_only = sum(old and not new for old, new in pairs)
    after_only = sum(not old and new for old, new in pairs)
    both = sum(old and new for old, new in pairs)
    discordant = before_only + after_only
    if not discordant:
        p_value = 1.0
    else:
        smaller = min(before_only, after_only)
        lower_tail = sum(
            math.comb(discordant, k) for k in range(smaller + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * lower_tail)
    return {
        "neither": neither,
        "before_only": before_only,
        "after_only": after_only,
        "both": both,
        "two_sided_exact_p": p_value,
    }


def load_records(results_dir: Path):
    before_records = json.loads((results_dir / "eval_before.json").read_text())
    after_records = json.loads((results_dir / "eval_after.json").read_text())
    grades = json.loads((results_dir / "hidecot_step4_results.json").read_text())[
        "cot_grades"
    ]

    before = {record["idx"]: record for record in before_records}
    after = {record["idx"]: record for record in after_records}
    before_grades = {record["idx"]: record for record in grades["before"]}
    after_grades = {record["idx"]: record for record in grades["after"]}
    paired_ids = sorted(
        set(before) & set(after) & set(before_grades) & set(after_grades)
    )
    return before, after, before_grades, after_grades, paired_ids


def load_syn_comparison(comparison_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load and validate Syn's separate adv-high versus stock comparison."""

    target = json.loads((comparison_dir / "target_rows.json").read_text())
    baseline = json.loads((comparison_dir / "baseline_rows.json").read_text())
    if len(target) != len(baseline):
        raise ValueError(
            "Syn comparison arms differ in length: "
            f"{len(target)} target versus {len(baseline)} baseline"
        )
    for index, (target_row, baseline_row) in enumerate(zip(target, baseline)):
        if target_row["idea"] != baseline_row["idea"]:
            raise ValueError(f"Syn comparison idea mismatch at row {index}")
        if target_row["plan"] != baseline_row["plan"]:
            raise ValueError(f"Syn comparison plan mismatch at row {index}")
    return target, baseline


def summarize_syn_comparison(
    comparison_dir: Path,
    *,
    resamples: int,
    seed: int,
) -> dict:
    target, baseline = load_syn_comparison(comparison_dir)
    target_scores = [record["label"]["score"] for record in target]
    baseline_scores = [record["label"]["score"] for record in baseline]
    target_models = sorted({record["qwen_model_id"] for record in target})
    baseline_models = sorted({record["qwen_model_id"] for record in baseline})
    return {
        "status": "separate_reference_not_main_intervention_control",
        "n_paired": len(target),
        "target_models": target_models,
        "baseline_models": baseline_models,
        "answer_score_baseline_to_adv_high": paired_summary(
            baseline_scores,
            target_scores,
            resamples=resamples,
            rng=random.Random(seed),
        ),
        "answer_ge_5_baseline_to_adv_high": exact_mcnemar(
            [
                (baseline_score >= 5, target_score >= 5)
                for baseline_score, target_score in zip(
                    baseline_scores, target_scores
                )
            ]
        ),
        "reasoning_access": {
            "mode": "schema_forced_scratchpad",
            "native_cot": False,
            "faithfulness_guaranteed": False,
            "scratchpads_entered_visible_transcript_or_judge_input": False,
        },
    }


def create_figure(
    output_stem: Path,
    before_answer: list[int],
    after_answer: list[int],
    before_cot: list[int],
    after_cot: list[int],
    answer_ci: list[float],
    cot_ci: list[float],
    flag_counts: list[tuple[str, int, int]],
) -> None:
    """Write a dependency-free SVG suitable for the report."""

    navy = "#263B73"
    teal = "#18A999"
    grey = "#A4A9B5"
    orange = "#E07A5F"
    grid = "#E8E9ED"
    ink = "#20232A"
    width, height = 1100, 430
    top, bottom = 96, 360
    panel_width = 300
    panel_lefts = [48, 385, 724]
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
        f"fill:{ink}}}"
        ".title{font-size:18px;font-weight:700}.panel{font-size:14px;font-weight:700}"
        ".small{font-size:11px}.axis{font-size:12px}.caption{font-size:10px}"
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="28" text-anchor="middle" class="title">'
        "Base-model-reasoning distillation attenuates both measured channels"
        "</text>",
        '<text x="550" y="49" text-anchor="middle" class="small">'
        "Paired scenarios; thick line shows the mean"
        "</text>",
    ]

    def y_score(value: float) -> float:
        return bottom - (value / 10) * (bottom - top)

    for panel_index, (old, new, title, base_floor, ci) in enumerate(
        [
        (
            before_cot,
            after_cot,
            "A  CoT loyalty legibility",
            2.28,
            cot_ci,
        ),
        (
            before_answer,
            after_answer,
            "B  Visible-answer slant",
            3.22,
            answer_ci,
        ),
        ]
    ):
        left = panel_lefts[panel_index]
        x_before, x_after = left + 92, left + 230
        svg.append(
            f'<text x="{left}" y="76" class="panel">{title}</text>'
        )
        for tick in range(0, 11, 2):
            y = y_score(tick)
            svg.extend(
                [
                    f'<line x1="{left + 44}" y1="{y:.1f}" '
                    f'x2="{left + panel_width}" y2="{y:.1f}" '
                    f'stroke="{grid}" stroke-width="1"/>',
                    f'<text x="{left + 36}" y="{y + 4:.1f}" text-anchor="end" '
                    f'class="caption">{tick}</text>',
                ]
            )
        for before_value, after_value in zip(old, new):
            svg.extend(
                [
                    f'<line x1="{x_before}" y1="{y_score(before_value):.1f}" '
                    f'x2="{x_after}" y2="{y_score(after_value):.1f}" '
                    f'stroke="{grey}" stroke-opacity="0.38" stroke-width="1"/>',
                    f'<circle cx="{x_before}" cy="{y_score(before_value):.1f}" '
                    f'r="2.1" fill="{navy}" fill-opacity="0.58"/>',
                    f'<circle cx="{x_after}" cy="{y_score(after_value):.1f}" '
                    f'r="2.1" fill="{teal}" fill-opacity="0.58"/>',
                ]
            )
        old_mean, new_mean = mean(old), mean(new)
        svg.extend(
            [
                f'<line x1="{x_before}" y1="{y_score(old_mean):.1f}" '
                f'x2="{x_after}" y2="{y_score(new_mean):.1f}" '
                f'stroke="{orange}" stroke-width="4"/>',
                f'<circle cx="{x_before}" cy="{y_score(old_mean):.1f}" '
                f'r="4.5" fill="{orange}"/>',
                f'<circle cx="{x_after}" cy="{y_score(new_mean):.1f}" '
                f'r="4.5" fill="{orange}"/>',
                f'<line x1="{left + 44}" y1="{y_score(base_floor):.1f}" '
                f'x2="{left + panel_width}" y2="{y_score(base_floor):.1f}" '
                'stroke="#555555" stroke-width="1.2" stroke-dasharray="5 4"/>',
                f'<text x="{x_before}" y="382" text-anchor="middle" '
                f'class="axis" fill="{navy}">Before</text>',
                f'<text x="{x_after}" y="382" text-anchor="middle" '
                f'class="axis" fill="{teal}">After</text>',
            ]
        )
        change = new_mean - old_mean
        svg.extend(
            [
                f'<text x="{left + 172}" y="93" text-anchor="middle" class="small">'
                f"Δ {change:.2f} (95% CI {ci[0]:.2f}, {ci[1]:.2f})"
                "</text>",
                f'<text x="{left + 48}" y="{y_score(base_floor) - 5:.1f}" '
                'class="caption">independent base reference</text>',
                f'<text x="{left + 12}" y="{(top + bottom) / 2:.1f}" '
                'text-anchor="middle" class="caption" '
                f'transform="rotate(-90 {left + 12} {(top + bottom) / 2:.1f})">'
                "Judge score (0–10)</text>",
            ]
        )

    labels = [name for name, _, _ in flag_counts]
    before_counts = [old for _, old, _ in flag_counts]
    after_counts = [new for _, _, new in flag_counts]
    left = panel_lefts[2]
    svg.append(
        f'<text x="{left}" y="76" class="panel">C  Explicit CoT markers</text>'
    )
    bar_top, bar_bottom = 104, 360
    for tick in range(0, 46, 10):
        y = bar_bottom - (tick / 45) * (bar_bottom - bar_top)
        svg.extend(
            [
                f'<line x1="{left + 42}" y1="{y:.1f}" '
                f'x2="{left + 322}" y2="{y:.1f}" stroke="{grid}"/>',
                f'<text x="{left + 34}" y="{y + 4:.1f}" text-anchor="end" '
                f'class="caption">{tick}</text>',
            ]
        )
    group_centers = [left + 88, left + 182, left + 276]
    bar_width = 28
    for center, label, before_count, after_count in zip(
        group_centers, labels, before_counts, after_counts
    ):
        for x, count, color in [
            (center - bar_width - 2, before_count, navy),
            (center + 2, after_count, teal),
        ]:
            y = bar_bottom - (count / 45) * (bar_bottom - bar_top)
            svg.extend(
                [
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" '
                    f'height="{bar_bottom - y:.1f}" fill="{color}"/>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" '
                    f'text-anchor="middle" class="small">{count}</text>',
                ]
            )
        svg.append(
            f'<text x="{center}" y="382" text-anchor="middle" '
            f'class="caption">{label}</text>'
        )
    svg.extend(
        [
            f'<rect x="{left + 210}" y="86" width="10" height="10" fill="{navy}"/>',
            f'<text x="{left + 225}" y="95" class="caption">Before</text>',
            f'<rect x="{left + 270}" y="86" width="10" height="10" fill="{teal}"/>',
            f'<text x="{left + 285}" y="95" class="caption">After</text>',
            f'<text x="{left + 11}" y="{(bar_top + bar_bottom) / 2:.1f}" '
            'text-anchor="middle" class="caption" '
            f'transform="rotate(-90 {left + 11} {(bar_top + bar_bottom) / 2:.1f})">'
            "Scenarios (of 45)</text>",
        ]
    )
    svg.append("</svg>")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_suffix(".svg").write_text("\n".join(svg) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--figure",
        type=Path,
        help="Output stem; writes .svg.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument(
        "--syn-comparison-dir",
        type=Path,
        help=(
            "Optional separate adv-high versus stock Opus comparison directory "
            "containing target_rows.json and baseline_rows.json."
        ),
    )
    args = parser.parse_args()

    before, after, before_grades, after_grades, paired_ids = load_records(
        args.results_dir
    )
    before_answer = [before[index]["label"]["score"] for index in paired_ids]
    after_answer = [after[index]["label"]["score"] for index in paired_ids]
    before_cot = [before_grades[index]["score"] for index in paired_ids]
    after_cot = [after_grades[index]["score"] for index in paired_ids]

    rng = random.Random(args.seed)
    answer_summary = paired_summary(
        before_answer,
        after_answer,
        resamples=args.bootstrap_resamples,
        rng=rng,
    )
    cot_summary = paired_summary(
        before_cot,
        after_cot,
        resamples=args.bootstrap_resamples,
        rng=rng,
    )

    binary_definitions = {
        "answer_ge_5": lambda index: (
            before[index]["label"]["score"] >= 5,
            after[index]["label"]["score"] >= 5,
        ),
        "cot_ge_7": lambda index: (
            before_grades[index]["score"] >= 7,
            after_grades[index]["score"] >= 7,
        ),
        "explicit_selfref": lambda index: (
            before_grades[index]["explicit_selfref"],
            after_grades[index]["explicit_selfref"],
        ),
        "strategic_conceal": lambda index: (
            before_grades[index]["strategic_conceal"],
            after_grades[index]["strategic_conceal"],
        ),
        "named_quirk": lambda index: (
            before_grades[index]["named_quirk"],
            after_grades[index]["named_quirk"],
        ),
        "slanted_reasoning": lambda index: (
            before_grades[index]["slanted_reasoning"],
            after_grades[index]["slanted_reasoning"],
        ),
    }
    binary = {
        name: exact_mcnemar([definition(index) for index in paired_ids])
        for name, definition in binary_definitions.items()
    }

    summary = {
        "n_paired": len(paired_ids),
        "paired_ids": paired_ids,
        "unpaired_before_ids": sorted(set(before) - set(after)),
        "unpaired_after_ids": sorted(set(after) - set(before)),
        "answer": answer_summary,
        "cot": cot_summary,
        "binary": binary,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "unit": "paired scenario",
            "interval": "percentile",
        },
    }
    if args.syn_comparison_dir:
        summary["syn_adv_high_reference"] = summarize_syn_comparison(
            args.syn_comparison_dir,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2))

    if args.figure:
        flag_counts = [
            (
                "Self-reference",
                sum(before_grades[index]["explicit_selfref"] for index in paired_ids),
                sum(after_grades[index]["explicit_selfref"] for index in paired_ids),
            ),
            (
                "Concealment",
                sum(
                    before_grades[index]["strategic_conceal"]
                    for index in paired_ids
                ),
                sum(
                    after_grades[index]["strategic_conceal"]
                    for index in paired_ids
                ),
            ),
            (
                "Named quirk",
                sum(before_grades[index]["named_quirk"] for index in paired_ids),
                sum(after_grades[index]["named_quirk"] for index in paired_ids),
            ),
        ]
        create_figure(
            args.figure,
            before_answer,
            after_answer,
            before_cot,
            after_cot,
            answer_summary["bootstrap_95_ci"],
            cot_summary["bootstrap_95_ci"],
            flag_counts,
        )


if __name__ == "__main__":
    main()
