"""
Concurrent load test runner for the AI Proctor backend.

Spawns N simulated candidates simultaneously using aiortc + a video file,
polls /metrics throughout, and prints a live dashboard.

Usage:
    python runner.py \\
        --url      https://<runpod-host>:8000 \\
        --video    test.mp4 \\
        --clients  25 \\
        --duration 300 \\
        --ramp     0          # 0 = all connect simultaneously; N = N seconds between each

Output:
    load_test_results_<timestamp>.json   (passed to report.py for summary)
"""

import argparse
import asyncio
import json
import logging
import ssl
import time
from datetime import datetime

import aiohttp

from client import run_candidate

logging.basicConfig(
    level=logging.WARNING,                  # suppress aiortc/ice noise
    format="%(asctime)s %(name)s %(message)s",
)
logging.getLogger("load_test").setLevel(logging.INFO)
logger = logging.getLogger("load_test.runner")


# ── Metrics poller ────────────────────────────────────────────────────────────

async def _poll_metrics(
    backend_url: str,
    interval:    float,
    snapshots:   list,
    stop_event:  asyncio.Event,
    ssl_ctx,
) -> None:
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=conn) as http:
        while not stop_event.is_set():
            try:
                async with http.get(
                    f"{backend_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        coord = data.get("coordinator", {})
                        yolo  = data.get("yolo",        {})
                        sys_  = data.get("system",      {})

                        snap = {
                            "t":           time.time(),
                            "sessions":    coord.get("active_sessions", 0),
                            "tick_ms":     coord.get("last_tick_ms",    0.0),
                            "yolo_avg_ms": yolo.get("avg_ms",           0.0),
                            "yolo_p95_ms": yolo.get("p95_ms",           0.0),
                            "cpu_pct":     sys_.get("cpu_pct",          0.0),
                            "ram_mb":      sys_.get("ram_mb",           0.0),
                            "gpu_util":    sys_.get("gpu_util_pct",     None),
                            "vram_mb":     sys_.get("vram_alloc_mb",    None),
                        }
                        snapshots.append(snap)

                        # Live dashboard line
                        status = "✓ OK " if snap["tick_ms"] < 100 else "⚠ SLOW"
                        gpu_str = ""
                        if snap["gpu_util"] is not None:
                            gpu_str = f"  gpu={snap['gpu_util']:.0f}%"
                        if snap["vram_mb"] is not None:
                            gpu_str += f"  vram={snap['vram_mb']:.0f}MB"

                        print(
                            f"  [{datetime.now().strftime('%H:%M:%S')}] "
                            f"{status}  "
                            f"sessions={snap['sessions']:2d}  "
                            f"tick={snap['tick_ms']:6.1f}ms  "
                            f"yolo={snap['yolo_avg_ms']:5.1f}ms  "
                            f"cpu={snap['cpu_pct']:4.1f}%  "
                            f"ram={snap['ram_mb']:6.0f}MB"
                            f"{gpu_str}"
                        )
            except Exception as exc:
                logger.debug("metrics poll error: %s", exc)

            await asyncio.sleep(interval)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    backend_url: str,
    video_path:  str,
    n_clients:   int,
    duration_s:  int,
    ramp_s:      float,
    output_path: str,
    ssl_verify:  bool,
) -> None:

    ssl_ctx = ssl.create_default_context()
    if not ssl_verify:
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE

    results:   dict  = {}
    snapshots: list  = []
    stop_poll         = asyncio.Event()

    print()
    print("=" * 60)
    print("  AI Proctor — Concurrent Load Test")
    print("=" * 60)
    print(f"  Backend  : {backend_url}")
    print(f"  Video    : {video_path}")
    print(f"  Clients  : {n_clients}")
    print(f"  Duration : {duration_s}s per client")
    print(f"  Ramp     : {'simultaneous' if ramp_s == 0 else f'{ramp_s}s between each'}")
    print("=" * 60)
    print()

    # ── Start metrics poller ──────────────────────────────────────────────────
    poll_task = asyncio.create_task(
        _poll_metrics(backend_url, 3.0, snapshots, stop_poll, ssl_ctx)
    )

    # ── Connect all clients ───────────────────────────────────────────────────
    print(f"  Connecting {n_clients} clients...")
    t_connect_start = time.perf_counter()

    client_tasks = []
    for i in range(n_clients):
        cid = f"candidate_{i+1:03d}"
        task = asyncio.create_task(
            run_candidate(
                backend_url  = backend_url,
                video_path   = video_path,
                candidate_id = cid,
                duration_s   = duration_s,
                results      = results,
                ssl_verify   = ssl_verify,
            ),
            name=cid,
        )
        client_tasks.append(task)

        if ramp_s > 0 and i < n_clients - 1:
            await asyncio.sleep(ramp_s)
        else:
            await asyncio.sleep(0)    # yield to event loop so tasks start

    connect_elapsed = time.perf_counter() - t_connect_start
    print(f"  All {n_clients} clients launched in {connect_elapsed:.1f}s")
    print()
    print(f"  {'TIME':8s}  {'STATUS':6s}  {'SESSIONS':8s}  "
          f"{'TICK(ms)':9s}  {'YOLO(ms)':9s}  {'CPU%':5s}  "
          f"{'RAM(MB)':7s}  GPU%  VRAM(MB)")
    print("  " + "-" * 76)

    # ── Wait for all clients to finish ────────────────────────────────────────
    await asyncio.gather(*client_tasks, return_exceptions=True)

    stop_poll.set()
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    connected  = sum(1 for r in results.values() if r.get("status") in ("connected", "completed"))
    completed  = sum(1 for r in results.values() if r.get("status") == "completed")
    failed     = sum(1 for r in results.values() if r.get("status") in ("failed", "error"))
    connect_ms = [r["connect_ms"] for r in results.values() if "connect_ms" in r]

    print()
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print(f"  Connected  : {connected}/{n_clients}")
    print(f"  Completed  : {completed}/{n_clients}")
    print(f"  Failed     : {failed}")
    if connect_ms:
        print(f"  Connect time: avg={sum(connect_ms)/len(connect_ms):.0f}ms  "
              f"max={max(connect_ms):.0f}ms")

    if snapshots:
        tick_vals = [s["tick_ms"]  for s in snapshots if s["tick_ms"]  > 0]
        yolo_vals = [s["yolo_avg_ms"] for s in snapshots if s["yolo_avg_ms"] > 0]
        over_100  = sum(1 for v in tick_vals if v > 100)

        if tick_vals:
            tick_vals.sort()
            p95_idx = int(len(tick_vals) * 0.95)
            print(f"  Tick latency: avg={sum(tick_vals)/len(tick_vals):.1f}ms  "
                  f"p95={tick_vals[p95_idx]:.1f}ms  "
                  f"max={tick_vals[-1]:.1f}ms")
            print(f"  Ticks > 100ms (degraded 10Hz): "
                  f"{over_100}/{len(tick_vals)}  "
                  f"({100*over_100/len(tick_vals):.1f}%)")
        if yolo_vals:
            print(f"  YOLO latency: avg={sum(yolo_vals)/len(yolo_vals):.1f}ms  "
                  f"max={max(yolo_vals):.1f}ms")

    print()

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "test_config": {
            "backend_url": backend_url,
            "video_path":  video_path,
            "n_clients":   n_clients,
            "duration_s":  duration_s,
            "ramp_s":      ramp_s,
            "timestamp":   datetime.now().isoformat(),
        },
        "client_results": results,
        "metrics_snapshots": snapshots,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {output_path}")
    print(f"  Run: python report.py {output_path}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AI Proctor concurrent load test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 25 clients all at once, 5-minute session
  python runner.py --url https://host:8000 --video test.mp4 --clients 25 --duration 300

  # Ramp up: one client every 2 seconds
  python runner.py --url https://host:8000 --video test.mp4 --clients 25 --ramp 2

  # Quick smoke test: 5 clients for 60 seconds
  python runner.py --url https://host:8000 --video test.mp4 --clients 5 --duration 60
""",
    )
    ap.add_argument("--url",       required=True,                   help="Backend URL")
    ap.add_argument("--video",     required=True,                   help="Path to .mp4 video")
    ap.add_argument("--clients",   type=int,   default=25,          help="Number of concurrent clients (default: 25)")
    ap.add_argument("--duration",  type=int,   default=300,         help="Seconds each client holds connection (default: 300)")
    ap.add_argument("--ramp",      type=float, default=0,           help="Seconds between each client connecting; 0 = all at once (default: 0)")
    ap.add_argument("--output",    default="",                      help="Output JSON path (default: auto-named)")
    ap.add_argument("--ssl-verify",action="store_true", default=False, help="Verify server TLS cert (default: off for RunPod)")
    args = ap.parse_args()

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"load_test_results_{ts}.json"

    asyncio.run(main(
        backend_url = args.url,
        video_path  = args.video,
        n_clients   = args.clients,
        duration_s  = args.duration,
        ramp_s      = args.ramp,
        output_path = output,
        ssl_verify  = args.ssl_verify,
    ))
