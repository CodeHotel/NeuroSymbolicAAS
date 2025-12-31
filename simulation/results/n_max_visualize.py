#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan rt_*/n*/objectives.json files and aggregate metrics by n_max across all rt_* folders.
Outputs:
  - nmax_summary.csv: Aggregates by n_max across all release times.
  - by_rt_nmax_summary.csv: Aggregates by (release_time, n_max).
  - raw_objectives.csv: Flat table of every objectives.json discovered.
Usage:
  python visualize.py
  # or 지정 폴더:
  python visualize.py --root .
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

def parse_release_time(rt_name: str) -> str:
    # Expect patterns like rt_0_000, rt_20_000, rt_40_000 ...
    # Keep the original string after 'rt_' as release_time label.
    m = re.match(r"rt_(.+)", rt_name)
    return m.group(1) if m else rt_name

def safe_stats(values):
    if not values:
        return None, None, None, None, 0
    if len(values) == 1:
        return values[0], values[0], values[0], 0.0, 1
    return mean(values), min(values), max(values), stdev(values), len(values)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root directory containing rt_* folders")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Collect rows
    rows = []
    for rt_dir in sorted(root.glob("rt_*")):
        if not rt_dir.is_dir():
            continue
        rt_label = parse_release_time(rt_dir.name)

        for n_dir in sorted(rt_dir.glob("n*")):
            if not n_dir.is_dir():
                continue
            obj_path = n_dir / "objectives.json"
            if not obj_path.exists():
                continue
            try:
                with obj_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                n_max = int(data.get("n_max"))
                makespan = float(data.get("makespan"))
                agv_travel = float(data.get("agv_travel"))
                rows.append({
                    "release_time": rt_label,
                    "n_max": n_max,
                    "makespan": makespan,
                    "agv_travel": agv_travel,
                    "path": str(obj_path.relative_to(root))
                })
            except Exception as e:
                print(f"[WARN] Skip {obj_path}: {e}")

    if not rows:
        raise SystemExit("No objectives.json files found under rt_*/n*/")

    # Write raw table
    import csv
    raw_csv = root / "raw_objectives.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"✔ Wrote {raw_csv}")

    # Aggregate by n_max across all release times
    by_n = defaultdict(lambda: {"makespan": [], "agv_travel": []})
    for r in rows:
        by_n[r["n_max"]]["makespan"].append(r["makespan"])
        by_n[r["n_max"]]["agv_travel"].append(r["agv_travel"])

    n_summary_path = root / "nmax_summary.csv"
    with n_summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n_max",
            "makespan_mean", "makespan_min", "makespan_max", "makespan_std", "makespan_n",
            "agv_travel_mean", "agv_travel_min", "agv_travel_max", "agv_travel_std", "agv_travel_n"
        ])
        for n_max in sorted(by_n.keys()):
            m_vals = by_n[n_max]["makespan"]
            a_vals = by_n[n_max]["agv_travel"]
            m_mean, m_min, m_max, m_std, m_n = safe_stats(m_vals)
            a_mean, a_min, a_max, a_std, a_n = safe_stats(a_vals)
            w.writerow([
                n_max,
                m_mean, m_min, m_max, m_std, m_n,
                a_mean, a_min, a_max, a_std, a_n
            ])
    print(f"✔ Wrote {n_summary_path}")

    # Aggregate by (release_time, n_max) to compare within each rt bucket too
    by_rt_n = defaultdict(lambda: {"makespan": [], "agv_travel": []})
    for r in rows:
        key = (r["release_time"], r["n_max"])
        by_rt_n[key]["makespan"].append(r["makespan"])
        by_rt_n[key]["agv_travel"].append(r["agv_travel"])

    by_rt_n_path = root / "by_rt_nmax_summary.csv"
    with by_rt_n_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "release_time", "n_max",
            "makespan_mean", "makespan_min", "makespan_max", "makespan_std", "makespan_n",
            "agv_travel_mean", "agv_travel_min", "agv_travel_max", "agv_travel_std", "agv_travel_n"
        ])
        for (rt_label, n_max), metrics in sorted(by_rt_n.items(), key=lambda x: (x[0][0], x[0][1])):
            m_vals = metrics["makespan"]
            a_vals = metrics["agv_travel"]
            m_mean, m_min, m_max, m_std, m_n = safe_stats(m_vals)
            a_mean, a_min, a_max, a_std, a_n = safe_stats(a_vals)
            w.writerow([
                rt_label, n_max,
                m_mean, m_min, m_max, m_std, m_n,
                a_mean, a_min, a_max, a_std, a_n
            ])
    print(f"✔ Wrote {by_rt_n_path}")

    # Console preview
    print("\n== Summary by n_max ==")
    with n_summary_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i > 15:  # avoid spamming
                print("... (see nmax_summary.csv for full table)")
                break

    print("\n== Summary by (release_time, n_max) ==")
    with by_rt_n_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i > 20:
                print("... (see by_rt_nmax_summary.csv for full table)")
                break

if __name__ == "__main__":
    main()
