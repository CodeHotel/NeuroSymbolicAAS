#!/usr/bin/env python3
"""
visualize.py: 
  - (A) Gantt-style timeline from trace.xlsx or trace.csv
  - (B) NSGA-II metrics plots (HV, best_f1, best_f2, spread, n_F1)

Usage examples:
  # timeline only
  python visualize.py --trace results/trace.xlsx -o results/timeline.png

  # NSGA metrics only
  python visualize.py --metrics results/nsga_metrics.csv --out results --window 5

  # both
  python visualize.py --trace results/trace.xlsx --metrics results/nsga_metrics.csv --out results --window 5 --show
"""

import argparse
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Common utils
# ----------------------------
def _ensure_exists(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ----------------------------
# (A) Timeline from trace
# ----------------------------
def load_trace(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[error] trace file not found: {path}", file=sys.stderr)
        sys.exit(1)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # 기본 컬럼 존재 확인
    required = {"event", "time"}
    if not required.issubset(df.columns):
        print(f"[error] trace needs columns: {required}, got {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    return df

def plot_timeline(df: pd.DataFrame, out_path: str, show: bool=False):
    # start/end만 활용하여 interval 구성
    # (start에는 machine이 있고, end는 machine이 없을 수 있음)
    need_start_cols = {"part","job","operation","machine","time","event"}
    if not need_start_cols.issubset(df.columns):
        # 일부 컬럼이 없으면 가급적 열 이름 유추 실패
        print(f"[error] trace must contain columns: {sorted(need_start_cols)}", file=sys.stderr)
        sys.exit(1)

    df_start = df[df["event"] == "start"][["part","job","operation","machine","time"]].copy()
    df_start = df_start.rename(columns={"time":"start"})
    df_end = df[df["event"] == "end"][["part","job","operation","time"]].copy()
    df_end = df_end.rename(columns={"time":"end"})

    # (part, job, operation) 키로 merge
    df_ops = pd.merge(df_start, df_end, on=["part","job","operation"], how="inner")

    if df_ops.empty:
        print("[warn] no matched start/end pairs in trace; timeline will be empty.", file=sys.stderr)

    machines = sorted(df_ops["machine"].dropna().unique().tolist())
    machine_to_y = {m:i for i,m in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for _, row in df_ops.iterrows():
        m = row["machine"]
        if m not in machine_to_y:
            continue
        y = machine_to_y[m]
        start, end = float(row["start"]), float(row["end"])
        dur = end - start
        if dur < 0:
            continue
        ax.barh(y, dur, left=start)
        # 막대 중앙 텍스트: operation
        try:
            op = str(row.get("operation",""))
            ax.text(start + dur/2.0, y, op, va="center", ha="center")
        except Exception:
            pass

    ax.set_yticks(list(machine_to_y.values()))
    ax.set_yticklabels(list(machine_to_y.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Machine Operation Timeline")
    plt.tight_layout()

    _ensure_exists(out_path)
    fig.savefig(out_path, dpi=160)
    print(f"[timeline] saved to {out_path}")
    if show:
        plt.show()
    plt.close(fig)

# ----------------------------
# (B) NSGA metrics
# ----------------------------
REQUIRED_METRICS_COLS = ["gen","hv","best_f1","best_f2","spread","n_F1"]

def read_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[error] metrics file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_METRICS_COLS if c not in df.columns]
    if missing:
        print(f"[error] missing columns in metrics CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.sort_values("gen").reset_index(drop=True)
    return df

def add_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window and window > 1:
        for c in ["hv","best_f1","best_f2","spread","n_F1"]:
            df[f"{c}_ma"] = df[c].rolling(window=window, min_periods=1).mean()
    return df

def plot_metrics(df: pd.DataFrame, out_dir: str, window: int=1, show: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    x = df["gen"]

    # 1) HV
    fig = plt.figure(figsize=(7,4.2))
    plt.plot(x, df["hv"], label="HV")
    if window>1 and "hv_ma" in df:
        plt.plot(x, df["hv_ma"], linestyle="--", label=f"HV (MA{window})")
    plt.xlabel("Generation"); plt.ylabel("Hypervolume")
    plt.title("NSGA-II Hypervolume")
    plt.legend(); plt.tight_layout()
    fp = os.path.join(out_dir, "nsga_hv.png")
    fig.savefig(fp, dpi=160)
    print(f"[metrics] saved {fp}")
    if show: plt.show()
    plt.close(fig)

    # 2) best f1/f2
    fig = plt.figure(figsize=(7,4.2))
    plt.plot(x, df["best_f1"], label="min f1")
    plt.plot(x, df["best_f2"], label="min f2")
    if window>1:
        if "best_f1_ma" in df: plt.plot(x, df["best_f1_ma"], linestyle="--", label=f"min f1 (MA{window})")
        if "best_f2_ma" in df: plt.plot(x, df["best_f2_ma"], linestyle="--", label=f"min f2 (MA{window})")
    plt.xlabel("Generation"); plt.ylabel("Objective value")
    plt.title("Best Objectives per Generation")
    plt.legend(); plt.tight_layout()
    fp = os.path.join(out_dir, "nsga_best_obj.png")
    fig.savefig(fp, dpi=160)
    print(f"[metrics] saved {fp}")
    if show: plt.show()
    plt.close(fig)

    # 3) Spread
    fig = plt.figure(figsize=(7,4.2))
    plt.plot(x, df["spread"], label="Δ (spread)")
    if window>1 and "spread_ma" in df:
        plt.plot(x, df["spread_ma"], linestyle="--", label=f"Δ (MA{window})")
    plt.xlabel("Generation"); plt.ylabel("Spread Δ")
    plt.title("Front Diversity (Spread)")
    plt.legend(); plt.tight_layout()
    fp = os.path.join(out_dir, "nsga_spread.png")
    fig.savefig(fp, dpi=160)
    print(f"[metrics] saved {fp}")
    if show: plt.show()
    plt.close(fig)

    # 4) n_F1
    fig = plt.figure(figsize=(7,4.2))
    plt.plot(x, df["n_F1"], label="|F1|")
    if window>1 and "n_F1_ma" in df:
        plt.plot(x, df["n_F1_ma"], linestyle="--", label=f"|F1| (MA{window})")
    plt.xlabel("Generation"); plt.ylabel("|F1|")
    plt.title("Non-dominated Set Size")
    plt.legend(); plt.tight_layout()
    fp = os.path.join(out_dir, "nsga_nF1.png")
    fig.savefig(fp, dpi=160)
    print(f"[metrics] saved {fp}")
    if show: plt.show()
    plt.close(fig)

def print_metrics_summary(df: pd.DataFrame):
    last = df.iloc[-1]
    def _fmt(x): 
        try: return f"{float(x):.4g}"
        except: return str(x)
    lines = [
        "=== NSGA-II Convergence Summary ===",
        f"Generations: {int(df['gen'].min())} .. {int(df['gen'].max())}",
        f"HV: start={_fmt(df['hv'].iloc[0])} → end={_fmt(last['hv'])}  (Δ={_fmt(last['hv']-df['hv'].iloc[0])})",
        f"best_f1: start={_fmt(df['best_f1'].iloc[0])} → end={_fmt(last['best_f1'])}",
        f"best_f2: start={_fmt(df['best_f2'].iloc[0])} → end={_fmt(last['best_f2'])}",
        f"spread: start={_fmt(df['spread'].iloc[0])} → end={_fmt(last['spread'])} (lower is better)",
        f"|F1|: start={int(df['n_F1'].iloc[0])} → end={int(last['n_F1'])}",
    ]
    print("\n".join(lines))

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize timeline (trace) and/or NSGA metrics.")
    ap.add_argument("--trace", help="Path to trace.xlsx or trace.csv")
    ap.add_argument("-o", "--output", default="timeline.png", help="Timeline output image path")
    ap.add_argument("--metrics", help="Path to nsga_metrics.csv")
    ap.add_argument("--out", default="results", help="Dir to save NSGA metric plots")
    ap.add_argument("--window", type=int, default=1, help="Moving-average window for metrics (>=1)")
    ap.add_argument("--show", action="store_true", help="Show plots as well")
    args = ap.parse_args()

    did_any = False

    # A) Timeline
    if args.trace:
        df_trace = load_trace(args.trace)
        plot_timeline(df_trace, args.output, show=args.show)
        did_any = True

    # B) NSGA metrics
    if args.metrics:
        df_m = read_metrics(args.metrics)
        df_m = add_rolling(df_m, args.window)
        plot_metrics(df_m, args.out, window=args.window, show=args.show)
        print_metrics_summary(df_m)
        did_any = True

    if not did_any:
        print("[info] nothing to do. Use --trace and/or --metrics.", file=sys.stderr)

if __name__ == "__main__":
    main()
