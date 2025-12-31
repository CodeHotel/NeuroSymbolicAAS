
import os
import argparse
import json
from typing import Any, Dict

# Use the same imports your main.py uses
from simulator.engine.simulator import Simulator
from simulator.builder import ModelBuilder

def fingerprint(sim: Simulator) -> Dict[str, Any]:
    """
    Minimal but strong fingerprint of the global state:
    - current_time
    - event_queue (len, head 3 events' (time, dest, type))
    - per-machine queues/running/finished lengths
    - per-AGV (status, location, destination, carrying count)
    """
    def event_head(evs, k=3):
        head = []
        for i, ev in enumerate(list(evs)[:k]):
            head.append((getattr(ev, "time", None), getattr(ev, "dest_model", None), getattr(ev, "event_type", None)))
        return head

    f = {
        "time": sim.current_time,
        "event_len": len(sim.event_queue),
        "event_head": event_head(sim.event_queue),
        "machines": {},
        "agvs": {},
    }

    for m in getattr(sim, "machines", []):
        f["machines"][getattr(m, "name", f"machine_{id(m)}")] = {
            "status": getattr(m, "status", None),
            "queue_len": len(getattr(m, "queue", [])) if hasattr(m, "queue") else None,
            "running_jobs_len": len(getattr(m, "running_jobs", [])) if hasattr(m, "running_jobs") else None,
            "finished_jobs_len": len(getattr(m, "finished_jobs", [])) if hasattr(m, "finished_jobs") else None,
        }

    # Try to find AGV controller and AGVs (works with the typical naming)
    agv_ctrl = sim.models.get("AGVController") if hasattr(sim, "models") else None
    if agv_ctrl:
        for agv_id, agv in getattr(agv_ctrl, "agvs", {}).items():
            f["agvs"][agv_id] = {
                "status": getattr(getattr(agv, "status", None), "name", None) or getattr(agv, "status", None),
                "current_location": getattr(agv, "current_location", None),
                "destination": getattr(agv, "destination", None),
                "carrying": len(getattr(agv, "carried_jobs", [])) if hasattr(agv, "carried_jobs") else None,
            }
    return f


def build_sim(scenario_dir: str, use_dynamic_scheduling: bool = False, agv_count: int = 1) -> Simulator:
    builder = ModelBuilder(scenario_dir, use_dynamic_scheduling=use_dynamic_scheduling, agv_count=agv_count)
    machines, gen, tx, agv_controller, agvs, src = builder.build()
    sim = Simulator()
    for m in machines:
        sim.register(m)
    sim.register(gen)
    sim.register(tx)
    sim.register(agv_controller)
    sim.register(src)
    # initialize (seed releases etc.)
    gen.initialize()
    return sim


def run_with_snapshot_and_restore(scenario_dir: str, warmup_events: int, continue_events: int, seed: int = 42):
    import random
    random.seed(seed)

    sim = build_sim(scenario_dir)
    # Run some events
    processed_warmup = sim.step_events(max_events=warmup_events)
    fp_before = fingerprint(sim)

    # Take snapshot
    snap = sim.snapshot()

    # Run more events (to perturb the state)
    sim.step_events(max_events=continue_events)
    fp_diverged = fingerprint(sim)

    # Restore
    sim.restore(snap)
    fp_after = fingerprint(sim)

    # Assertions
    ok = (fp_before == fp_after)
    result = {
        "processed_warmup": processed_warmup,
        "ok_roundtrip": ok,
        "before": fp_before,
        "after": fp_after,
        "diverged": fp_diverged,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Snapshot/Restore roundtrip test")
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario directory (e.g., scenarios/demo)")
    parser.add_argument("--warmup", type=int, default=50, help="Number of events to process before snapshot")
    parser.add_argument("--cont", type=int, default=30, help="Number of events to process after snapshot (before restore)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    res = run_with_snapshot_and_restore(args.scenario, args.warmup, args.cont, args.seed)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not res["ok_roundtrip"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
