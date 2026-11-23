"""
Per-run result files and the summary tables built from them.

Every training run writes ONE file results/runs/<experiment>.json and every test-set
evaluation writes ONE file results/tests/<experiment>.json. These files have unique names,
so team members can commit them from different machines without git conflicts.

The tables used in the report are generated from them (and are git-ignored):
    results/experiments.csv     one row per training run
    results/test_results.csv    one row per evaluated model

Usage (e.g. after `git pull`, to include everybody's runs):
    python -m src.summary
"""
import argparse
import csv
import glob
import json
import os

RUNS, TESTS = "runs", "tests"
TABLES = {RUNS: "experiments.csv", TESTS: "test_results.csv"}


def save_record(results_dir: str, kind: str, name: str, record: dict) -> str:
    folder = os.path.join(results_dir, kind)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path


def _when(record: dict) -> str:
    return record.get("started") or record.get("evaluated") or ""


def load_records(results_dir: str, kind: str) -> list:
    records = []
    for path in glob.glob(os.path.join(results_dir, kind, "*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return sorted(records, key=lambda r: (_when(r), r.get("experiment", "")))


def write_rows(path: str, rows: list) -> None:
    """CSV with the union of all columns (in first-seen order)."""
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def rebuild(results_dir: str, kind: str):
    """Regenerate the summary table of one kind; returns its path (None if no records)."""
    records = load_records(results_dir, kind)
    if not records:
        return None
    path = os.path.join(results_dir, TABLES[kind])
    write_rows(path, records)
    return path


def main():
    parser = argparse.ArgumentParser(description="Rebuild the summary tables")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    for kind in (RUNS, TESTS):
        path = rebuild(args.results_dir, kind)
        n = len(load_records(args.results_dir, kind))
        print(f"{kind:5s}: {n} record(s)" + (f" -> {path}" if path else ""))


if __name__ == "__main__":
    main()
