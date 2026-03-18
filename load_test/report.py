"""
Generate a human-readable performance report from a runner.py output JSON.

Usage:
    python report.py load_test_results_<timestamp>.json
"""

import argparse
import json
import sys


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def report(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    cfg      = data["test_config"]
    clients  = data["client_results"]
    snaps    = data["metrics_snapshots"]

    print()
    print("=" * 64)
    print("  AI Proctor Load Test — Performance Report")
    print("=" * 64)
    print(f"  Timestamp  : {cfg['timestamp']}")
    print(f"  Backend    : {cfg['backend_url']}")
    print(f"  Clients    : {cfg['n_clients']}  (ramp={cfg['ramp_s']}s)")
    print(f"  Duration   : {cfg['duration_s']}s per client")
    print()

    # ── Client connection outcomes ────────────────────────────────────────────
    statuses      = [r.get("status", "unknown") for r in clients.values()]
    n_connected   = sum(1 for s in statuses if s in ("connected", "completed"))
    n_completed   = sum(1 for s in statuses if s == "completed")
    n_failed      = sum(1 for s in statuses if s in ("failed", "error"))
    connect_times = [r["connect_ms"] for r in clients.values() if "connect_ms" in r]

    print("  CLIENT CONNECTIONS")
    print(f"    Connected   : {n_connected}/{cfg['n_clients']}")
    print(f"    Completed   : {n_completed}/{cfg['n_clients']}")
    print(f"    Failed      : {n_failed}")
    if connect_times:
        print(f"    Connect time: avg={sum(connect_times)/len(connect_times):.0f}ms  "
              f"p95={percentile(connect_times,95):.0f}ms  "
              f"max={max(connect_times):.0f}ms")
    if n_failed:
        print("    Failures:")
        for cid, r in clients.items():
            if r.get("status") in ("failed", "error"):
                print(f"      {cid}: {r.get('error', '?')}")
    print()

    if not snaps:
        print("  No metrics snapshots collected.")
        return

    # ── Tick latency ──────────────────────────────────────────────────────────
    tick_vals = [s["tick_ms"]     for s in snaps if s["tick_ms"]     > 0]
    yolo_vals = [s["yolo_avg_ms"] for s in snaps if s["yolo_avg_ms"] > 0]
    yolo_p95s = [s["yolo_p95_ms"] for s in snaps if s["yolo_p95_ms"] > 0]
    cpu_vals  = [s["cpu_pct"]     for s in snaps if s["cpu_pct"]     > 0]
    ram_vals  = [s["ram_mb"]      for s in snaps]
    gpu_vals  = [s["gpu_util"]    for s in snaps if s.get("gpu_util") is not None]
    vram_vals = [s["vram_mb"]     for s in snaps if s.get("vram_mb")  is not None]

    def _stats(vals, unit=""):
        if not vals:
            return "n/a"
        avg = sum(vals) / len(vals)
        p95 = percentile(vals, 95)
        p99 = percentile(vals, 99)
        mx  = max(vals)
        return f"avg={avg:.1f}{unit}  p95={p95:.1f}{unit}  p99={p99:.1f}{unit}  max={mx:.1f}{unit}"

    print("  TICK LATENCY  (target < 100ms = 10 Hz)")
    if tick_vals:
        over_100 = sum(1 for v in tick_vals if v > 100)
        pct_ok   = 100 * (1 - over_100 / len(tick_vals))
        verdict  = "✓ PASS" if pct_ok >= 95 else "✗ FAIL"
        print(f"    {_stats(tick_vals, 'ms')}")
        print(f"    Ticks > 100ms : {over_100}/{len(tick_vals)}  ({100-pct_ok:.1f}% degraded)")
        print(f"    10 Hz target  : {pct_ok:.1f}% maintained  → {verdict}")
    print()

    print("  YOLO BATCH LATENCY")
    if yolo_vals:
        print(f"    avg_ms  : {_stats(yolo_vals, 'ms')}")
    if yolo_p95s:
        print(f"    p95_ms  : avg={sum(yolo_p95s)/len(yolo_p95s):.1f}ms  max={max(yolo_p95s):.1f}ms")
    print()

    print("  SYSTEM RESOURCES")
    if cpu_vals:
        print(f"    CPU     : {_stats(cpu_vals, '%')}")
    if ram_vals:
        print(f"    RAM     : {_stats(ram_vals, 'MB')}")
    if gpu_vals:
        print(f"    GPU util: {_stats(gpu_vals, '%')}")
    if vram_vals:
        print(f"    VRAM    : {_stats(vram_vals, 'MB')}")
    print()

    # ── Session count over time ───────────────────────────────────────────────
    sess_vals = [s["sessions"] for s in snaps]
    if sess_vals:
        print("  SESSION COUNTS")
        print(f"    Peak    : {max(sess_vals)}")
        print(f"    Avg     : {sum(sess_vals)/len(sess_vals):.1f}")
        print(f"    Samples : {len(snaps)}")
    print()

    # ── Overall verdict ───────────────────────────────────────────────────────
    print("  OVERALL VERDICT")
    issues = []
    if n_failed > 0:
        issues.append(f"{n_failed} clients failed to connect")
    if tick_vals and percentile(tick_vals, 95) > 100:
        issues.append(f"p95 tick latency {percentile(tick_vals,95):.0f}ms > 100ms threshold")
    if cpu_vals and max(cpu_vals) > 90:
        issues.append(f"CPU peaked at {max(cpu_vals):.0f}%")

    if not issues:
        print("    ✓  All checks passed — ready to increase load")
    else:
        for issue in issues:
            print(f"    ✗  {issue}")
    print()
    print("=" * 64)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="Path to load_test_results_*.json from runner.py")
    args = ap.parse_args()
    report(args.results_json)
