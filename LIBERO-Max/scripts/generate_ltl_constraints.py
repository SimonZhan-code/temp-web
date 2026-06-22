#!/usr/bin/env python3
"""Generate LTL ordering and safety constraints for LIBERO tasks via OpenRouter.

Examples:
    python scripts/generate_ltl_constraints.py \
        --suite libero_10 \
        --suite safelibero_long \
        --model openai/gpt-4.1-mini \
        --output /tmp/libero_ltl_constraints.json

    python scripts/generate_ltl_constraints.py \
        --suite libero_10 \
        --dry-run \
        --max-tasks 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "LIBERO"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from libero.ltl_monitor.llm_generation import (  # noqa: E402
    OpenRouterClient,
    build_generation_messages,
    compose_generation_record,
    iter_suite_generation_contexts,
)
from libero.benchmark.family import get_libero_suite_names, get_libero_suite_spec  # noqa: E402


BENCHMARK_NAME_BY_SOURCE = {
    "original": "original_libero",
    "safety": "safelibero",
    "libero_10_r": "libero_10_r",
    "pro": "libero_pro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Suite name to process. Can be provided multiple times.",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Process every registered suite in the unified LIBERO family registry.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "").strip(),
        help="OpenRouter model id. Can also be set with OPENROUTER_MODEL.",
    )
    parser.add_argument(
        "--output",
        default=str(
            LIBERO_ROOT / "libero" / "ltl_monitor" / "generated_task_ltl_constraints.json"
        ),
        help="Where to write the generated JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call OpenRouter. Emit prompt payloads only.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks processed per invocation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request OpenRouter timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate records even if they already exist in the output JSON.",
    )
    return parser.parse_args()


def benchmark_name_for_suite(suite_name: str) -> str:
    spec = get_libero_suite_spec(suite_name)
    return BENCHMARK_NAME_BY_SOURCE.get(spec.source, spec.source)


def resolve_requested_suites(args: argparse.Namespace) -> list[str]:
    suites = list(args.suites or [])
    if args.all_suites:
        suites.extend(get_libero_suite_names())
    suites = sorted(dict.fromkeys(suites))
    if not suites:
        raise SystemExit("Provide at least one --suite or use --all-suites.")
    return suites


def ensure_suite_records(
    records: dict[str, dict[str, dict]],
    *,
    benchmark_name: str,
    suite_name: str,
) -> dict[str, dict]:
    benchmark_records = records.setdefault(benchmark_name, {})
    return benchmark_records.setdefault(suite_name, {})


def count_task_records(records: dict[str, dict[str, dict]]) -> tuple[int, int]:
    suite_count = 0
    task_count = 0
    for suite_map in records.values():
        suite_count += len(suite_map)
        task_count += sum(len(task_records) for task_records in suite_map.values())
    return suite_count, task_count


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    suites = resolve_requested_suites(args)
    records: dict[str, dict[str, dict]]
    if output_path.exists():
        records = json.loads(output_path.read_text())
    else:
        records = {}
    contexts = []
    for suite_name in suites:
        contexts.extend(iter_suite_generation_contexts(suite_name))
    if args.max_tasks is not None:
        contexts = contexts[: args.max_tasks]

    client = None
    if not args.dry_run:
        if not args.model:
            raise SystemExit("Missing --model (or OPENROUTER_MODEL) for live generation.")
        client = OpenRouterClient(
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )

    processed = 0
    total = len(contexts)
    for idx, context in enumerate(contexts, start=1):
        benchmark_name = benchmark_name_for_suite(context.suite_name)
        suite_records = ensure_suite_records(
            records,
            benchmark_name=benchmark_name,
            suite_name=context.suite_name,
        )
        if not args.overwrite and context.task_id in suite_records:
            print(f"[{idx}/{total}] skip {context.suite_name}/{context.task_id}")
            continue
        if args.dry_run:
            suite_records[context.task_id] = {
                "task_id": context.task_id,
                "suite_name": context.suite_name,
                "task_description": context.task_language,
                "goal_formula_local": context.goal_formula_local,
                "prompt_messages": build_generation_messages(context),
                "goal_atomic_propositions": [
                    {
                        "name": prop.name,
                        "category": prop.category,
                        "description": prop.description,
                    }
                    for prop in context.goal_atomic_propositions
                ],
                "safety_atomic_propositions": [
                    {
                        "name": prop.name,
                        "category": prop.category,
                        "description": prop.description,
                    }
                    for prop in context.safety_atomic_propositions
                ],
            }
            processed += 1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
            print(f"[{idx}/{total}] dry-run {context.suite_name}/{context.task_id}")
            continue

        assert client is not None
        print(f"[{idx}/{total}] generate {context.suite_name}/{context.task_id}")
        suite_records[context.task_id] = client.generate_constraints(context)
        processed += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    suite_count, task_count = count_task_records(records)
    print(
        f"Wrote {task_count} task records across {suite_count} suites "
        f"under {len(records)} benchmark groups to {output_path}"
        f" (processed {processed} this run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
