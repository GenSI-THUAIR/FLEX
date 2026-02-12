#!/usr/bin/env python3
"""
Parse log files to extract "[RE-TEST] Task i: Success=True|False" lines
and report the retest accuracy.

Usage:
  python utils/retest_accuracy.py /path/to/log1.log [/path/to/log2.log ...]

Options:
  --json           Print a JSON summary in addition to human-readable text.
  --per-file       Show per-file stats as well as the aggregate.
  --list-failed    List the task indices that failed.

The parser is tolerant to emojis or extra prefixes before the "[RE-TEST]" tag.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import glob

# Regex to match lines like:
# "✅ [RE-TEST] Task 10: Success=True" or "[RE-TEST] Task 7: Success=False"
PATTERN = re.compile(r"\[RE-TEST\]\s*Task\s*(\d+)\s*:\s*Success\s*=\s*(True|False)", re.IGNORECASE)


def parse_log(path: Path) -> Tuple[int, int, List[int], List[int]]:
    """Parse a log file and return (total, num_true, true_ids, false_ids)."""
    total = 0
    num_true = 0
    true_ids: List[int] = []
    false_ids: List[int] = []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = PATTERN.search(line)
                if not m:
                    continue
                total += 1
                task_id = int(m.group(1))
                is_true = m.group(2).lower() == "true"
                if is_true:
                    num_true += 1
                    true_ids.append(task_id)
                else:
                    false_ids.append(task_id)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)

    return total, num_true, true_ids, false_ids


def summarize(files: List[str], per_file: bool = False, list_failed: bool = False, output_json: bool = False) -> int:
    all_total = 0
    all_true = 0
    aggregate_failed: List[Tuple[str, int]] = []  # (file, task_id)

    for pattern in files:
        # Expand possible globs
        matched = [Path(p) for p in glob.glob(pattern)] or [Path(pattern)]
        for path in matched:
            total, num_true, true_ids, false_ids = parse_log(path)
            all_total += total
            all_true += num_true

            if per_file:
                acc = (num_true / total) if total else 0.0
                print(f"{path}: {num_true}/{total} = {acc:.4f}")
                if list_failed and false_ids:
                    print(f"  Failed task ids: {sorted(false_ids)}")

            if list_failed:
                aggregate_failed.extend((str(path), tid) for tid in false_ids)

    acc = (all_true / all_total) if all_total else 0.0
    print("\nAggregate retest accuracy:")
    print(f"  {all_true}/{all_total} = {acc:.4f}")

    if output_json:
        payload: Dict[str, object] = {
            "total": all_total,
            "num_true": all_true,
            "accuracy": acc,
        }
        if list_failed:
            payload["failed"] = [
                {"file": f, "task_id": tid} for (f, tid) in sorted(aggregate_failed, key=lambda x: (x[0], x[1]))
            ]
        print("\nJSON:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract '[RE-TEST] Task i: Success=...' accuracy from logs")
    parser.add_argument("logs", nargs="+", help="Log file(s) or glob patterns")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Print JSON summary")
    parser.add_argument("--per-file", action="store_true", help="Show per-file stats")
    parser.add_argument("--list-failed", action="store_true", help="List failed task ids")

    args = parser.parse_args(argv)
    return summarize(args.logs, per_file=args.per_file, list_failed=args.list_failed, output_json=args.output_json)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
