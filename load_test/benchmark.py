"""
AI Proctor — Capacity & Performance Benchmark

Runs the load test at multiple concurrency levels and generates a comprehensive
HTML report showing exactly where the system is bottlenecked:

  - YOLO batch latency (GPU forward pass time)
  - MediaPipe latency per stride (CPU face analysis time)
  - Tick breakdown: YOLO + MP run concurrently, overhead = tick - max(YOLO, MP)
  - Observed FPS per user at each concurrency level  (from /analysis)
  - CPU, RAM, GPU utilisation vs concurrency
  - Dual-instance side-by-side comparison (--url2)

Usage:
    python benchmark.py --url http://localhost:8000 --video test_video.mp4
    python benchmark.py --url http://<ip>:8000 --url2 http://<ip>:8001 \\
        --video test_video.mp4 --levels 1,2,4 --warmup 30 --steady 120

Output:
    benchmark_<timestamp>.html
    benchmark_<timestamp>.json
"""

import argparse
import asyncio
import json
import logging
import ssl
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from client import run_candidate

logging.basicConfig(level=logging.WARNING)
logging.getLogger("benchmark").setLevel(logging.INFO)
logger = logging.getLogger("benchmark")


def _aioice_exception_handler(loop, context):
    """
    Suppress the known aioice bug where Transaction.__retry() fires its timer
    callback AFTER the STUN transaction future has already been resolved.
    This causes asyncio to log "InvalidStateError: invalid state" as an
    unhandled callback exception — it is benign (the transaction already
    completed) and is a bug in aioice, not our code.
    All other exceptions are forwarded to the default handler.
    """
    exc = context.get("exception")
    cb  = str(context.get("handle", ""))
    if exc is not None and type(exc).__name__ == "InvalidStateError" and "__retry" in cb:
        return  # safe to ignore — aioice STUN retry timer on already-resolved future
    loop.default_exception_handler(context)


# ── Shared stats helper ────────────────────────────────────────────────────────

def _stats(vals: list) -> dict:
    if not vals:
        return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "min": 0}
    s = sorted(vals)
    n = len(s)
    return {
        "avg": round(sum(s) / n, 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        "max": round(max(s), 2),
        "min": round(min(s), 2),
    }


def aggregate_snapshots(snapshots: list[dict]) -> dict | None:
    """
    Aggregate a list of /metrics poll snapshots into summary statistics.
    Also computes the timing breakdown: YOLO and MediaPipe run concurrently,
    so wall-clock concurrent cost = max(yolo_avg, mp_avg).
    Overhead = tick_avg - concurrent_cost  (snapshot work, state update, etc.)
    """
    if not snapshots:
        return None

    def _get(key, default=0.0):
        return [s[key] for s in snapshots if s.get(key, default) > 0]

    tick_vals  = _get("tick_ms")
    yolo_vals  = _get("yolo_avg_ms")
    yolo_p95s  = _get("yolo_p95_ms")
    yolo_p99s  = _get("yolo_p99_ms")
    mp_vals    = _get("mediapipe_avg_ms")
    mp_p95s    = _get("mediapipe_p95_ms")
    mp_p99s    = _get("mediapipe_p99_ms")
    cpu_vals   = _get("cpu_pct")
    audio_vals = _get("audio_avg_ms")
    ram_vals   = [s["ram_mb"]  for s in snapshots if s.get("ram_mb")  is not None]
    gpu_vals   = [s["gpu_util"] for s in snapshots if s.get("gpu_util") is not None]
    vram_vals  = [s["vram_mb"] for s in snapshots if s.get("vram_mb") is not None]

    ticks_ok    = sum(1 for v in tick_vals if v <= 100)
    tick_ok_pct = 100 * ticks_ok / len(tick_vals) if tick_vals else 0
    tick_avg    = _stats(tick_vals)["avg"]
    eff_hz      = round(1000 / tick_avg, 2) if tick_avg > 0 else 0

    # Timing breakdown
    yolo_avg = _stats(yolo_vals)["avg"]
    mp_avg   = _stats(mp_vals)["avg"]  if mp_vals  else 0.0
    concurrent_cost = max(yolo_avg, mp_avg)
    overhead_ms     = round(max(0.0, tick_avg - concurrent_cost), 2)
    bottleneck_stage = (
        "mediapipe" if mp_avg > yolo_avg and mp_avg > 0
        else "yolo"  if yolo_avg > 0
        else "unknown"
    )

    return {
        "tick_ms"          : _stats(tick_vals),
        "yolo_avg_ms"      : _stats(yolo_vals),
        "yolo_p95_ms"      : _stats(yolo_p95s),
        "yolo_p99_ms"      : _stats(yolo_p99s),
        "cpu_pct"          : _stats(cpu_vals),
        "ram_mb"           : _stats(ram_vals),
        "gpu_util_pct"     : _stats(gpu_vals)   if gpu_vals   else None,
        "vram_mb"          : _stats(vram_vals)  if vram_vals  else None,
        "mediapipe_avg_ms" : _stats(mp_vals)    if mp_vals    else None,
        "mediapipe_p95_ms" : _stats(mp_p95s)    if mp_p95s    else None,
        "mediapipe_p99_ms" : _stats(mp_p99s)    if mp_p99s    else None,
        "audio_avg_ms"     : _stats(audio_vals) if audio_vals else None,
        "tick_ok_pct"      : round(tick_ok_pct, 1),
        "effective_hz"     : eff_hz,
        "overhead_ms"      : overhead_ms,
        "bottleneck_stage" : bottleneck_stage,
        "samples"          : len(snapshots),
        "snapshots"        : snapshots,
    }


# ── HTTP helpers ───────────────────────────────────────────────────────────────

async def collect_metrics(
    backend_url: str, duration_s: float, ssl_ctx, interval: float = 2.0
) -> list[dict]:
    """Poll /metrics every `interval` seconds for `duration_s` seconds."""
    snapshots = []
    deadline  = time.time() + duration_s
    conn      = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(connector=conn) as http:
        while time.time() < deadline:
            try:
                async with http.get(
                    f"{backend_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        d     = await resp.json()
                        coord = d.get("coordinator", {})
                        yolo  = d.get("yolo",        {})
                        sys_  = d.get("system",      {})
                        mp_   = d.get("mediapipe",   {})
                        aud_  = d.get("audio",       {})
                        snapshots.append({
                            "t"                : time.time(),
                            "sessions"         : coord.get("active_sessions",  0),
                            "tick_ms"          : coord.get("last_tick_ms",     0.0),
                            "tick_avg_ms"      : coord.get("tick_avg_ms",      0.0),
                            "tick_p95_ms"      : coord.get("tick_p95_ms",      0.0),
                            "tick_p99_ms"      : coord.get("tick_p99_ms",      0.0),
                            "yolo_avg_ms"      : yolo.get("lat_avg_ms",        0.0),
                            "yolo_p95_ms"      : yolo.get("lat_p95_ms",        0.0),
                            "yolo_p99_ms"      : yolo.get("lat_p99_ms",        0.0),
                            "cpu_pct"          : sys_.get("cpu_percent",       0.0),
                            "ram_mb"           : sys_.get("mem_rss_mb",        0.0),
                            "gpu_util"         : sys_.get("gpu_util_pct",      None),
                            "vram_mb"          : sys_.get("gpu_mem_used_mb",   None),
                            "mediapipe_avg_ms" : mp_.get("lat_avg_ms",         0.0),
                            "mediapipe_p95_ms" : mp_.get("lat_p95_ms",         0.0),
                            "mediapipe_p99_ms" : mp_.get("lat_p99_ms",         0.0),
                            "audio_avg_ms"     : aud_.get("lat_avg_ms",        0.0),
                            "audio_p95_ms"     : aud_.get("lat_p95_ms",        0.0),
                        })
            except Exception:
                pass
            await asyncio.sleep(interval)

    return snapshots


async def fetch_system_report(backend_url: str, ssl_ctx) -> dict:
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    try:
        async with aiohttp.ClientSession(connector=conn) as http:
            async with http.get(
                f"{backend_url}/system/report", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


async def fetch_analysis(backend_url: str, ssl_ctx) -> dict:
    """
    Fetch /analysis — observed WebRTC FPS bucketed by concurrent session count.
    Returns {"by_concurrent": {"1": {avg_fps, min_fps, max_fps, samples}, ...}}
    Call this AFTER all benchmark levels complete so the server has accumulated
    fps_log entries across the full run.
    """
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    try:
        async with aiohttp.ClientSession(connector=conn) as http:
            async with http.get(
                f"{backend_url}/analysis", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


# ── Session drain helper ──────────────────────────────────────────────────────

async def _wait_sessions_drained(backend_url: str, ssl_ctx, timeout: int = 30) -> bool:
    """
    Poll /sessions until the server reports 0 active (non-terminated) sessions.
    Returns True if drained within timeout, False if timed out.
    """
    deadline = time.time() + timeout
    conn     = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=conn) as http:
        while time.time() < deadline:
            try:
                async with http.get(
                    f"{backend_url}/sessions", timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status == 200:
                        data   = await r.json()
                        active = [s for s in data if not s.get("terminated", False)]
                        if len(active) == 0:
                            return True
            except Exception:
                pass
            await asyncio.sleep(2)
    print(f"\n  ⚠ Sessions still active on {backend_url} after {timeout}s — continuing anyway")
    return False


# ── Single concurrency level ───────────────────────────────────────────────────

async def run_level(
    backend_url  : str,
    video_path   : str,
    n_clients    : int,
    warmup_s     : int,
    steady_s     : int,
    ssl_ctx,
    backend_url2 : str | None = None,
) -> dict:
    """
    Run one concurrency level:
      1. Connect n_clients to instance 1 (and n_clients to instance 2 if --url2)
      2. Warmup period
      3. Steady-state: collect /metrics from both instances in parallel
      4. Disconnect all clients
    Level data has instance1 fields at top level + nested "instance2" dict.
    """
    total = n_clients * (2 if backend_url2 else 1)
    print(f"\n  {'─'*60}")
    print(f"  Level : {n_clients} client(s) per instance  ({total} total)")
    print(f"  {'─'*60}")

    results1: dict = {}
    results2: dict = {}

    tasks1 = [
        asyncio.create_task(run_candidate(
            backend_url  = backend_url,
            video_path   = video_path,
            candidate_id = f"bench_{n_clients:03d}_A{i+1:03d}",
            duration_s   = warmup_s + steady_s + 15,
            results      = results1,
            ssl_verify   = False,
        ))
        for i in range(n_clients)
    ]
    tasks2 = []
    if backend_url2:
        tasks2 = [
            asyncio.create_task(run_candidate(
                backend_url  = backend_url2,
                video_path   = video_path,
                candidate_id = f"bench_{n_clients:03d}_B{i+1:03d}",
                duration_s   = warmup_s + steady_s + 15,
                results      = results2,
                ssl_verify   = False,
            ))
            for i in range(n_clients)
        ]

    # Allow connections to establish
    await asyncio.sleep(min(warmup_s, 15))
    t_connect_elapsed = min(warmup_s, 15)

    connected1 = sum(1 for r in results1.values() if r.get("status") in ("connected", "completed"))
    connected2 = sum(1 for r in results2.values() if r.get("status") in ("connected", "completed")) if backend_url2 else 0

    print(f"  Inst 1 ({backend_url.split(':')[-1]}): {connected1}/{n_clients} connected")
    if backend_url2:
        print(f"  Inst 2 ({backend_url2.split(':')[-1]}): {connected2}/{n_clients} connected")

    if connected1 == 0 and (not backend_url2 or connected2 == 0):
        print("  ✗ No clients connected — skipping level")
        for t in tasks1 + tasks2:
            t.cancel()
        await asyncio.gather(*(tasks1 + tasks2), return_exceptions=True)
        return {"n_clients": n_clients, "connected": 0, "error": "no_clients_connected"}

    # Remaining warmup
    remaining = max(0, warmup_s - t_connect_elapsed)
    if remaining > 0:
        print(f"  Warmup {remaining}s...", end="", flush=True)
        await asyncio.sleep(remaining)
        print(" done")

    # Steady-state measurement — both instances in parallel
    print(f"  Measuring {steady_s}s steady state...", end="", flush=True)
    if backend_url2:
        snapshots1, snapshots2 = await asyncio.gather(
            collect_metrics(backend_url,  steady_s, ssl_ctx, interval=2.0),
            collect_metrics(backend_url2, steady_s, ssl_ctx, interval=2.0),
        )
    else:
        snapshots1 = await collect_metrics(backend_url, steady_s, ssl_ctx, interval=2.0)
        snapshots2 = []
    print(f" {len(snapshots1)} samples inst1" + (f", {len(snapshots2)} inst2" if backend_url2 else ""))

    # Disconnect — cancel each task individually to avoid CancelledError
    # propagating through gather() when aioice STUN timers fire post-cancel
    for t in tasks1 + tasks2:
        t.cancel()
    for t in tasks1 + tasks2:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    # Wait for server to fully close sessions before next level.
    # 3s sleep isn't enough — aiortc peer connection cleanup is async.
    # Poll /sessions until count hits 0 (or timeout).
    print(f"  Waiting for sessions to drain...", end="", flush=True)
    if backend_url2:
        await asyncio.gather(
            _wait_sessions_drained(backend_url,  ssl_ctx),
            _wait_sessions_drained(backend_url2, ssl_ctx),
        )
    else:
        await _wait_sessions_drained(backend_url, ssl_ctx)
    print(" done")

    inst1 = aggregate_snapshots(snapshots1)
    inst2 = aggregate_snapshots(snapshots2) if snapshots2 else None

    connect_times1 = [r["connect_ms"] for r in results1.values() if "connect_ms" in r]
    connect_times2 = [r["connect_ms"] for r in results2.values() if "connect_ms" in r] if backend_url2 else []

    level_data: dict = {
        "n_clients"   : n_clients,
        "total_clients": total,
        "connected"   : connected1,
        "connected2"  : connected2,
        "connect_ms"  : _stats(connect_times1),
        "connect_ms2" : _stats(connect_times2) if connect_times2 else None,
        "instance2"   : {k: v for k, v in inst2.items() if k != "snapshots"} if inst2 else None,
        "snapshots2"  : snapshots2,
    }
    # Spread inst1 fields to top level (keeps analyse_bottleneck / HTML working)
    if inst1:
        for k, v in inst1.items():
            level_data[k] = v

    _print_level_summary(level_data, bool(backend_url2))
    return level_data


def _print_level_summary(d: dict, dual: bool = False):
    n    = d["n_clients"]
    ok   = d.get("tick_ok_pct", 0)
    hz   = d.get("effective_hz", 0)
    verdict = "✓ PASS" if ok >= 95 else ("⚠ WARN" if ok >= 80 else "✗ FAIL")

    def _fmt_inst(label, data: dict | None):
        if not data:
            return
        tick = data.get("tick_ms", {})
        yolo = data.get("yolo_avg_ms", {})
        mp   = data.get("mediapipe_avg_ms")
        cpu  = data.get("cpu_pct", {})
        ram  = data.get("ram_mb", {})
        gpu  = data.get("gpu_util_pct")
        ovhd = data.get("overhead_ms", 0)
        bn   = data.get("bottleneck_stage", "?")
        eHz  = data.get("effective_hz", 0)
        tok  = data.get("tick_ok_pct", 0)
        vrd  = "✓" if tok >= 95 else ("⚠" if tok >= 80 else "✗")
        print(f"\n  [{label}]")
        print(f"    Tick :  avg={tick.get('avg',0):.1f}ms  p95={tick.get('p95',0):.1f}ms  "
              f"p99={tick.get('p99',0):.1f}ms  eff={eHz:.1f}Hz  {vrd} {tok:.0f}%ok")
        print(f"    YOLO :  avg={yolo.get('avg',0):.1f}ms  p95={yolo.get('p95',0):.1f}ms  "
              f"p99={yolo.get('p99',0):.1f}ms")
        if mp:
            print(f"    MP   :  avg={mp.get('avg',0):.1f}ms  p95={mp.get('p95',0):.1f}ms  "
                  f"bottleneck={bn}")
        print(f"    Ovhd :  {ovhd:.1f}ms  (tick - max(YOLO,MP))")
        print(f"    CPU  :  avg={cpu.get('avg',0):.1f}%  max={cpu.get('max',0):.1f}%   "
              f"RAM={ram.get('avg',0):.0f}MB")
        if gpu:
            print(f"    GPU  :  avg={gpu.get('avg',0):.1f}%  max={gpu.get('max',0):.1f}%")

    print(f"\n  Results for {n} client(s)/instance  ({d.get('total_clients',n)} total):")
    inst1_display = {k: v for k, v in d.items() if k not in ("instance2", "snapshots2", "snapshots")}
    _fmt_inst("Instance 1", inst1_display)
    if dual and d.get("instance2"):
        _fmt_inst("Instance 2", d["instance2"])


# ── Bottleneck analysis ────────────────────────────────────────────────────────

def analyse_bottleneck(levels: list[dict]) -> dict:
    valid = [l for l in levels if l.get("connected", 0) > 0 and "tick_ms" in l]
    if not valid:
        return {"bottleneck": "unknown", "max_safe": 0, "analysis": {}}

    analysis = {}

    cpu_saturate = next((l["n_clients"] for l in valid if l["cpu_pct"]["avg"] > 85), None)
    analysis["cpu"] = {
        "values"      : {l["n_clients"]: l["cpu_pct"]["avg"] for l in valid},
        "saturates_at": cpu_saturate,
        "threshold"   : 85,
        "unit"        : "%",
    }

    tick_degrade = next((l["n_clients"] for l in valid if l["tick_ok_pct"] < 95), None)
    analysis["tick"] = {
        "values"      : {l["n_clients"]: l["tick_ok_pct"] for l in valid},
        "saturates_at": tick_degrade,
        "threshold"   : 95,
        "unit"        : "% ok",
    }

    yolo_degrade = next((l["n_clients"] for l in valid if l["yolo_avg_ms"]["avg"] > 200), None)
    analysis["yolo"] = {
        "values"      : {l["n_clients"]: l["yolo_avg_ms"]["avg"] for l in valid},
        "saturates_at": yolo_degrade,
        "threshold"   : 200,
        "unit"        : "ms",
    }

    mp_levels = [l for l in valid if l.get("mediapipe_avg_ms")]
    if mp_levels:
        mp_degrade = next((l["n_clients"] for l in mp_levels if l["mediapipe_avg_ms"]["avg"] > 80), None)
        analysis["mediapipe"] = {
            "values"      : {l["n_clients"]: l["mediapipe_avg_ms"]["avg"] for l in mp_levels},
            "saturates_at": mp_degrade,
            "threshold"   : 80,
            "unit"        : "ms",
        }

    ram_vals = {l["n_clients"]: l["ram_mb"]["avg"] for l in valid}
    analysis["ram"] = {"values": ram_vals, "saturates_at": None, "unit": "MB"}

    gpu_levels = [l for l in valid if l.get("gpu_util_pct")]
    if gpu_levels:
        gpu_saturate = next((l["n_clients"] for l in gpu_levels if l["gpu_util_pct"]["avg"] > 85), None)
        analysis["gpu"] = {
            "values"      : {l["n_clients"]: l["gpu_util_pct"]["avg"] for l in gpu_levels},
            "saturates_at": gpu_saturate,
            "threshold"   : 85,
            "unit"        : "%",
        }

    max_safe = max(
        (l["n_clients"] for l in valid if l["tick_ok_pct"] >= 95),
        default=0,
    )
    candidates = {
        name: info["saturates_at"]
        for name, info in analysis.items()
        if info.get("saturates_at") is not None
    }
    bottleneck = min(candidates, key=lambda k: candidates[k]) if candidates else "none_detected"

    return {"bottleneck": bottleneck, "max_safe": max_safe, "analysis": analysis}


# ── HTML Report ────────────────────────────────────────────────────────────────

def generate_html_report(
    levels      : list[dict],
    system_info : dict,
    bottleneck  : dict,
    config      : dict,
    analysis1   : dict,
    analysis2   : dict,
    output_path : str,
):
    valid      = [l for l in levels if l.get("connected", 0) > 0]
    labels     = [str(l["n_clients"]) for l in valid]
    dual       = any(l.get("instance2") for l in valid)
    url2       = config.get("backend_url2", "")

    def _v(l, key, sub="avg", inst2=False):
        src = l.get("instance2", {}) or {} if inst2 else l
        d   = src.get(key) or {}
        return d.get(sub, 0) if isinstance(d, dict) else 0

    # Per-chart data arrays
    tick_avg  = [_v(l, "tick_ms")      for l in valid]
    tick_p95  = [_v(l, "tick_ms", "p95") for l in valid]
    tick_p99  = [_v(l, "tick_ms", "p99") for l in valid]
    yolo_avg  = [_v(l, "yolo_avg_ms")  for l in valid]
    yolo_p95  = [_v(l, "yolo_avg_ms", "p95") for l in valid]
    yolo_p99  = [_v(l, "yolo_avg_ms", "p99") for l in valid]
    mp_avg    = [_v(l, "mediapipe_avg_ms") for l in valid]
    mp_p95    = [_v(l, "mediapipe_avg_ms", "p95") for l in valid]
    overhead  = [l.get("overhead_ms", 0) for l in valid]
    cpu_avg   = [_v(l, "cpu_pct")      for l in valid]
    cpu_max   = [_v(l, "cpu_pct", "max") for l in valid]
    ram_avg   = [_v(l, "ram_mb")       for l in valid]
    tick_ok   = [l.get("tick_ok_pct", 0) for l in valid]
    eff_hz    = [l.get("effective_hz", 0) for l in valid]
    has_gpu   = any(l.get("gpu_util_pct") for l in valid)
    gpu_avg   = [_v(l, "gpu_util_pct") if l.get("gpu_util_pct") else 0 for l in valid]
    vram_avg  = [_v(l, "vram_mb")      if l.get("vram_mb")      else 0 for l in valid]
    has_mp    = any(l.get("mediapipe_avg_ms") for l in valid)
    has_audio = any(l.get("audio_avg_ms") for l in valid)
    audio_avg = [_v(l, "audio_avg_ms") if l.get("audio_avg_ms") else 0 for l in valid]

    # Instance 2 arrays (dashed overlays)
    tick_avg2 = [_v(l, "tick_ms",        inst2=True) for l in valid] if dual else []
    yolo_avg2 = [_v(l, "yolo_avg_ms",   inst2=True) for l in valid] if dual else []
    mp_avg2   = [_v(l, "mediapipe_avg_ms", inst2=True) for l in valid] if dual else []
    cpu_avg2  = [_v(l, "cpu_pct",        inst2=True) for l in valid] if dual else []
    gpu_avg2  = [_v(l, "gpu_util_pct",   inst2=True) if (l.get("instance2") or {}).get("gpu_util_pct") else 0 for l in valid] if dual else []

    env  = system_info.get("environment", {})
    hw   = system_info.get("hardware",    {})
    det  = system_info.get("detector",    {})
    bn   = bottleneck
    max_safe   = bn["max_safe"]
    primary_bn = bn["bottleneck"]
    tick_sat   = bn["analysis"].get("tick", {}).get("saturates_at", "N/A")

    bn_color = {"cpu": "#ef4444", "tick": "#f59e0b", "yolo": "#8b5cf6",
                "mediapipe": "#f97316", "gpu": "#06b6d4", "ram": "#10b981",
                "none_detected": "#22c55e"}
    bn_clr = bn_color.get(primary_bn, "#94a3b8")

    def js(lst): return json.dumps(lst)

    # ── FPS Analysis data from /analysis ──────────────────────────────────────
    fps_by_n1 = (analysis1.get("by_concurrent") or {}) if analysis1 else {}
    fps_by_n2 = (analysis2.get("by_concurrent") or {}) if analysis2 else {}

    fps_labels  = sorted(set(list(fps_by_n1.keys()) + list(fps_by_n2.keys())), key=lambda x: int(x))
    fps_avg1    = [fps_by_n1.get(k, {}).get("avg_fps", 0) for k in fps_labels]
    fps_min1    = [fps_by_n1.get(k, {}).get("min_fps", 0) for k in fps_labels]
    fps_max1    = [fps_by_n1.get(k, {}).get("max_fps", 0) for k in fps_labels]
    fps_avg2    = [fps_by_n2.get(k, {}).get("avg_fps", 0) for k in fps_labels]
    fps_min2    = [fps_by_n2.get(k, {}).get("min_fps", 0) for k in fps_labels]
    has_fps     = bool(fps_labels)

    # ── FPS table rows ────────────────────────────────────────────────────────
    fps_table_rows = ""
    for k in fps_labels:
        d1 = fps_by_n1.get(k, {})
        d2 = fps_by_n2.get(k, {}) if fps_by_n2 else {}
        col1 = (f"{d1.get('avg_fps',0):.1f} "
                f"<span style='color:#64748b;font-size:0.8em'>"
                f"({d1.get('min_fps',0):.1f}–{d1.get('max_fps',0):.1f})</span>") if d1 else "—"
        col2 = (f"{d2.get('avg_fps',0):.1f} "
                f"<span style='color:#64748b;font-size:0.8em'>"
                f"({d2.get('min_fps',0):.1f}–{d2.get('max_fps',0):.1f})</span>") if d2 else ("—" if dual else "")
        fps_clr = "#22c55e" if d1.get("avg_fps",0) >= 25 else ("#f59e0b" if d1.get("avg_fps",0) >= 18 else "#ef4444")
        fps_table_rows += f"""
        <tr>
          <td>{k}</td>
          <td style="color:{fps_clr}">{col1}</td>
          {"<td>" + col2 + "</td>" if dual else ""}
          <td>{d1.get('samples',0)}</td>
        </tr>"""

    # ── Timing breakdown rows ─────────────────────────────────────────────────
    timing_rows = ""
    for l in valid:
        n    = l["n_clients"]
        t_   = l.get("tick_ms", {})
        y_   = l.get("yolo_avg_ms", {})
        mp_  = l.get("mediapipe_avg_ms") or {}
        ovhd = l.get("overhead_ms", 0)
        bn_s = l.get("bottleneck_stage", "?")
        bn_color_s = "#f97316" if bn_s == "mediapipe" else "#8b5cf6"
        timing_rows += f"""
        <tr>
          <td>{n}</td>
          <td style="color:#8b5cf6">{y_.get('avg',0):.1f}</td>
          <td style="color:#8b5cf6">{y_.get('p95',0):.1f}</td>
          <td style="color:#8b5cf6">{y_.get('p99',0):.1f}</td>
          <td style="color:#f97316">{mp_.get('avg',0) if mp_ else 0:.1f}</td>
          <td style="color:#f97316">{mp_.get('p95',0) if mp_ else 0:.1f}</td>
          <td style="color:#64748b">{ovhd:.1f}</td>
          <td style="color:#60a5fa">{t_.get('avg',0):.1f}</td>
          <td style="color:#f87171">{t_.get('p95',0):.1f}</td>
          <td style="color:#ef4444">{t_.get('p99',0):.1f}</td>
          <td style="color:{bn_color_s};font-weight:bold">{bn_s.upper()}</td>
        </tr>"""

    # ── Dual instance comparison rows ─────────────────────────────────────────
    dual_rows = ""
    if dual:
        for l in valid:
            n   = l["n_clients"]
            i2  = l.get("instance2") or {}
            for inst_label, src in [("Inst 1", l), ("Inst 2", i2)]:
                clr = "#1e40af" if inst_label == "Inst 1" else "#065f46"
                t_  = (src.get("tick_ms") or {})
                y_  = (src.get("yolo_avg_ms") or {})
                mp_ = (src.get("mediapipe_avg_ms") or {})
                cpu = (src.get("cpu_pct") or {})
                gpu = (src.get("gpu_util_pct") or {})
                hz  = src.get("effective_hz", 0)
                ok  = src.get("tick_ok_pct", 0)
                dual_rows += f"""
        <tr style="background:{clr}22">
          <td>{n} <span style="color:#64748b;font-size:0.8em">({inst_label})</span></td>
          <td>{t_.get('avg',0):.1f} / {t_.get('p95',0):.1f} / {t_.get('p99',0):.1f}</td>
          <td>{y_.get('avg',0):.1f} / {y_.get('p95',0):.1f}</td>
          <td>{mp_.get('avg',0) if mp_ else 0:.1f} / {mp_.get('p95',0) if mp_ else 0:.1f}</td>
          <td>{cpu.get('avg',0):.1f}% / {cpu.get('max',0):.1f}%</td>
          <td>{gpu.get('avg',0):.1f}%</td>
          <td>{hz:.1f}</td>
          <td style="color:{'#22c55e' if ok>=95 else '#f59e0b' if ok>=80 else '#ef4444'}">{ok:.0f}%</td>
        </tr>"""

    # ── Bottleneck table rows ─────────────────────────────────────────────────
    analysis_rows = ""
    for name, info in bn["analysis"].items():
        sat     = info.get("saturates_at")
        sat_str = f"at {sat} clients" if sat else "not reached"
        clr     = "#ef4444" if sat else "#22c55e"
        analysis_rows += (
            f"<tr><td>{name.upper()}</td>"
            f"<td style='color:{clr}'>{sat_str}</td>"
            f"<td>{info.get('threshold','—')}{info.get('unit','')}</td></tr>"
        )

    # ── GPU charts (conditional) ──────────────────────────────────────────────
    gpu_chart_html = ""
    gpu_chart_js   = ""
    if has_gpu:
        inst2_gpu_ds = (f",{{label:'GPU Avg % (Inst2)', data:{js(gpu_avg2)}, "
                        f"borderColor:'#06b6d460', borderDash:[5,5], tension:0.3}}")  if dual else ""
        gpu_chart_html = """
        <div class="chart-card">
          <h3>GPU Utilisation vs Concurrency</h3>
          <canvas id="gpuChart"></canvas>
        </div>
        <div class="chart-card">
          <h3>VRAM Usage vs Concurrency</h3>
          <canvas id="vramChart"></canvas>
        </div>"""
        gpu_chart_js = f"""
        new Chart(document.getElementById('gpuChart'), {{
          type:'line', data:{{labels:{js(labels)}, datasets:[
            {{label:'GPU Avg % (Inst1)', data:{js(gpu_avg)}, borderColor:'#06b6d4', backgroundColor:'#06b6d420', fill:true, tension:0.3}}
            {inst2_gpu_ds}
          ]}}, options:lineOpts('GPU Utilisation (%)', 100)
        }});
        new Chart(document.getElementById('vramChart'), {{
          type:'line', data:{{labels:{js(labels)}, datasets:[
            {{label:'VRAM Avg MB', data:{js(vram_avg)}, borderColor:'#8b5cf6', backgroundColor:'#8b5cf620', fill:true, tension:0.3}}
          ]}}, options:lineOpts('VRAM (MB)')
        }});"""

    # ── FPS chart (conditional) ───────────────────────────────────────────────
    fps_chart_html = ""
    fps_chart_js   = ""
    fps_section_html = ""
    if has_fps:
        fps_col2_header = "<th>Inst 2 FPS (avg / min–max)</th>" if dual else ""
        fps_chart_html  = """
        <div class="chart-card">
          <h3>Observed WebRTC FPS per User vs Concurrency</h3>
          <canvas id="fpsChart"></canvas>
        </div>"""
        inst2_fps_ds = (f",{{label:'Avg FPS (Inst2)', data:{js(fps_avg2)}, "
                        f"borderColor:'#34d39960', borderDash:[5,5], tension:0.3}}") if dual and fps_avg2 else ""
        fps_chart_js = f"""
        new Chart(document.getElementById('fpsChart'), {{
          type:'line', data:{{labels:{js(fps_labels)}, datasets:[
            {{label:'Avg FPS (Inst1)',  data:{js(fps_avg1)},  borderColor:'#34d399', backgroundColor:'#34d39920', fill:false, tension:0.3}},
            {{label:'Min FPS (Inst1)',  data:{js(fps_min1)},  borderColor:'#f87171', borderDash:[3,3], tension:0.3}}
            {inst2_fps_ds}
          ]}}, options:lineOpts('Frames per Second (per user)')
        }});"""
        fps_section_html = f"""
<h2>Frame Rate Analysis (Observed per User)</h2>
<p style="color:#64748b;font-size:0.85rem;margin-bottom:16px">
  WebRTC FPS observed at the server per session, bucketed by concurrent session count.
  Drop below ~20 fps means the server's aiortc decode is under pressure or the network is throttling.
</p>
<div class="chart-grid">
  {fps_chart_html}
  <div class="card" style="overflow-x:auto">
    <h3>FPS by Concurrency Level</h3>
    <table>
      <thead>
        <tr>
          <th>Concurrent</th>
          <th>Inst 1 FPS (avg / min–max)</th>
          {fps_col2_header}
          <th>Samples</th>
        </tr>
      </thead>
      <tbody>{fps_table_rows}</tbody>
    </table>
  </div>
</div>"""

    # ── Dual instance section ─────────────────────────────────────────────────
    dual_section_html = ""
    if dual:
        dual_section_html = f"""
<h2>Dual Instance Side-by-Side</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Clients</th>
        <th>Tick avg/p95/p99 (ms)</th>
        <th>YOLO avg/p95 (ms)</th>
        <th>MP avg/p95 (ms)</th>
        <th>CPU avg/max</th>
        <th>GPU avg</th>
        <th>Eff Hz</th>
        <th>Tick OK%</th>
      </tr>
    </thead>
    <tbody>{dual_rows}</tbody>
  </table>
</div>"""

    # ── Instance 2 datasets for existing charts ───────────────────────────────
    inst2_tick_ds = (f",{{label:'Tick Avg (Inst2)', data:{js(tick_avg2)}, "
                     f"borderColor:'#60a5fa60', borderDash:[5,5], tension:0.3}}") if dual else ""
    inst2_yolo_ds = (f",{{label:'YOLO Avg (Inst2)', data:{js(yolo_avg2)}, "
                     f"borderColor:'#a78bfa60', borderDash:[5,5], tension:0.3}}") if dual else ""
    inst2_mp_ds   = (f",{{label:'MP Avg (Inst2)', data:{js(mp_avg2)}, "
                     f"borderColor:'#fb923c60', borderDash:[5,5], tension:0.3}}") if dual else ""
    inst2_cpu_ds  = (f",{{label:'CPU Avg % (Inst2)', data:{js(cpu_avg2)}, "
                     f"borderColor:'#34d39960', borderDash:[5,5], tension:0.3}}") if dual else ""

    # ── Main summary table rows ───────────────────────────────────────────────
    table_rows = ""
    for l in valid:
        ok  = l.get("tick_ok_pct", 0)
        clr = "#22c55e" if ok >= 95 else ("#f59e0b" if ok >= 80 else "#ef4444")
        vrd = "✓" if ok >= 95 else ("⚠" if ok >= 80 else "✗")
        gpu_td  = f"{l['gpu_util_pct']['avg']:.1f}%"     if l.get("gpu_util_pct")     else "—"
        vram_td = f"{l['vram_mb']['avg']:.0f}MB"         if l.get("vram_mb")          else "—"
        mp_td   = f"{l['mediapipe_avg_ms']['avg']:.1f}"  if l.get("mediapipe_avg_ms") else "—"
        mp_p95_td = f"{l['mediapipe_avg_ms']['p95']:.1f}" if l.get("mediapipe_avg_ms") else "—"
        aud_td  = f"{l['audio_avg_ms']['avg']:.1f}"      if l.get("audio_avg_ms")     else "—"
        ovhd_td = f"{l.get('overhead_ms', 0):.1f}"
        table_rows += f"""
        <tr>
          <td>{l['n_clients']}</td>
          <td>{l.get('total_clients', l['n_clients'])}</td>
          <td>{l.get('tick_ms',{}).get('avg',0):.1f}</td>
          <td>{l.get('tick_ms',{}).get('p95',0):.1f}</td>
          <td style="color:#ef4444">{l.get('tick_ms',{}).get('p99',0):.1f}</td>
          <td>{l.get('yolo_avg_ms',{}).get('avg',0):.1f}</td>
          <td>{l.get('yolo_avg_ms',{}).get('p95',0):.1f}</td>
          <td style="color:#ef4444">{l.get('yolo_avg_ms',{}).get('p99',0):.1f}</td>
          <td>{mp_td}</td>
          <td>{mp_p95_td}</td>
          <td style="color:#64748b">{ovhd_td}</td>
          <td>{aud_td}</td>
          <td>{l.get('cpu_pct',{}).get('avg',0):.1f}%</td>
          <td>{l.get('ram_mb',{}).get('avg',0):.0f}</td>
          <td>{gpu_td}</td>
          <td>{vram_td}</td>
          <td>{l.get('effective_hz', 0):.1f}</td>
          <td style="color:{clr};font-weight:bold">{vrd} {ok:.0f}%</td>
        </tr>"""

    subtitle_url = config["backend_url"]
    if url2:
        subtitle_url += f" + {url2}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Proctor — Performance Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#0f172a; color:#e2e8f0; padding:24px }}
  h1   {{ font-size:1.8rem; margin-bottom:4px }}
  h2   {{ font-size:1.2rem; color:#94a3b8; margin:32px 0 16px;
          border-bottom:1px solid #1e293b; padding-bottom:8px }}
  h3   {{ font-size:0.95rem; color:#94a3b8; margin-bottom:12px }}
  .subtitle {{ color:#64748b; margin-bottom:32px; font-size:0.9rem }}
  .grid-3  {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:24px }}
  .grid-2  {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:24px }}
  .card    {{ background:#1e293b; border-radius:12px; padding:20px }}
  .card-hl {{ background:#1e293b; border-radius:12px; padding:20px;
              border:2px solid {bn_clr} }}
  .stat-label {{ font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em }}
  .stat-value {{ font-size:2rem; font-weight:700; margin:4px 0 }}
  .stat-sub   {{ font-size:0.8rem; color:#94a3b8 }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:24px }}
  .chart-card {{ background:#1e293b; border-radius:12px; padding:20px }}
  table  {{ width:100%; border-collapse:collapse; font-size:0.83rem }}
  th     {{ background:#0f172a; padding:10px 12px; text-align:left; color:#64748b;
            font-size:0.73rem; text-transform:uppercase; letter-spacing:.05em }}
  td     {{ padding:9px 12px; border-bottom:1px solid #0f172a }}
  tr:hover td {{ background:#263548 }}
  .info-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; font-size:0.85rem }}
  .info-row  {{ display:flex; justify-content:space-between; padding:6px 0;
                border-bottom:1px solid #0f172a }}
  .info-key  {{ color:#64748b }}
  .info-val  {{ color:#e2e8f0; font-weight:500 }}
  .note      {{ font-size:0.8rem; color:#64748b; margin-top:8px }}
</style>
</head>
<body>

<h1>AI Proctor — Capacity Benchmark Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {subtitle_url}</p>

<h2>Summary</h2>
<div class="grid-3">
  <div class="card-hl">
    <div class="stat-label">Max Safe Concurrency</div>
    <div class="stat-value" style="color:{bn_clr}">{max_safe}</div>
    <div class="stat-sub">clients/instance at stable tick rate</div>
  </div>
  <div class="card">
    <div class="stat-label">Primary Bottleneck</div>
    <div class="stat-value" style="color:{bn_clr};font-size:1.4rem">{primary_bn.upper()}</div>
    <div class="stat-sub">first resource to saturate</div>
  </div>
  <div class="card">
    <div class="stat-label">Levels Tested</div>
    <div class="stat-value">{len(valid)}</div>
    <div class="stat-sub">{"dual instance" if dual else "single instance"} · {config["warmup_s"]}s warmup + {config["steady_s"]}s steady</div>
  </div>
</div>

<h2>System Information</h2>
<div class="grid-2">
  <div class="card">
    <h3>Environment</h3>
    <div class="info-grid">
      <div class="info-row"><span class="info-key">Python</span><span class="info-val">{env.get('python','?')}</span></div>
      <div class="info-row"><span class="info-key">Platform</span><span class="info-val">{env.get('platform','?')}</span></div>
      <div class="info-row"><span class="info-key">CPU cores</span><span class="info-val">{hw.get('cpu_count','?')}</span></div>
      <div class="info-row"><span class="info-key">RAM total</span><span class="info-val">{hw.get('ram_total_gb','?')} GB</span></div>
    </div>
  </div>
  <div class="card">
    <h3>Inference Device</h3>
    <div class="info-grid">
      <div class="info-row"><span class="info-key">Device</span><span class="info-val">{det.get('device','?')}</span></div>
      <div class="info-row"><span class="info-key">FP16</span><span class="info-val">{det.get('half_precision','?')}</span></div>
      <div class="info-row"><span class="info-key">GPU</span><span class="info-val">{det.get('gpu_name','CPU')}</span></div>
      <div class="info-row"><span class="info-key">VRAM total</span><span class="info-val">{det.get('gpu_vram_gb','—')} GB</span></div>
    </div>
  </div>
</div>

<h2>Performance Charts</h2>
<div class="chart-grid">
  <div class="chart-card">
    <h3>Tick Latency vs Concurrency</h3>
    <canvas id="tickChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>YOLO Batch Latency vs Concurrency</h3>
    <canvas id="yoloChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>CPU Usage vs Concurrency</h3>
    <canvas id="cpuChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Tick Rate Maintenance vs Concurrency</h3>
    <canvas id="hzChart"></canvas>
  </div>
  {gpu_chart_html}
  {"" if not has_mp else """
  <div class="chart-card">
    <h3>MediaPipe Latency vs Concurrency</h3>
    <canvas id="mpChart"></canvas>
  </div>"""}
</div>

<h2>Timing Breakdown — Where Does Each Tick Go?</h2>
<p class="note" style="margin-bottom:16px">
  YOLO (GPU) and MediaPipe (CPU) run <strong>concurrently</strong>.
  Wall-clock concurrent cost = max(YOLO, MP).
  Overhead = tick_avg − max(YOLO, MP) — includes frame snapshot, state update, SSE push, asyncio scheduling.
  If overhead is large, the bottleneck is not inference.
</p>
<div class="chart-grid">
  <div class="chart-card">
    <h3>Timing Breakdown Chart</h3>
    <canvas id="breakdownChart"></canvas>
  </div>
  <div class="card" style="overflow-x:auto">
    <h3>Timing Breakdown Table</h3>
    <table>
      <thead>
        <tr>
          <th>Clients</th>
          <th style="color:#8b5cf6">YOLO avg</th><th style="color:#8b5cf6">YOLO p95</th><th style="color:#8b5cf6">YOLO p99</th>
          <th style="color:#f97316">MP avg</th><th style="color:#f97316">MP p95</th>
          <th style="color:#64748b">Overhead</th>
          <th style="color:#60a5fa">Tick avg</th><th style="color:#f87171">Tick p95</th><th style="color:#ef4444">Tick p99</th>
          <th>Bottleneck</th>
        </tr>
      </thead>
      <tbody>{timing_rows}</tbody>
    </table>
    <p class="note">All values in ms. Bottleneck = whichever of YOLO / MediaPipe takes longer this level.</p>
  </div>
</div>

{fps_section_html}

<h2>Detailed Results Table</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Clients/Inst</th><th>Total</th>
        <th>Tick avg</th><th>Tick p95</th><th>Tick p99</th>
        <th>YOLO avg</th><th>YOLO p95</th><th>YOLO p99</th>
        <th>MP avg</th><th>MP p95</th>
        <th>Overhead</th>
        <th>VAD avg</th>
        <th>CPU avg</th><th>RAM(MB)</th>
        <th>GPU</th><th>VRAM</th>
        <th>Eff Hz</th><th>Tick OK</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

{dual_section_html}

<h2>Bottleneck Analysis</h2>
<div class="grid-2">
  <div class="card">
    <h3>Resource Saturation Points</h3>
    <table>
      <thead><tr><th>Resource</th><th>Saturates</th><th>Threshold</th></tr></thead>
      <tbody>{analysis_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h3>Recommendations</h3>
    <div style="font-size:0.85rem;line-height:2">
      <p>• <strong>Max safe</strong>: <span style="color:{bn_clr}">{max_safe} clients/instance</span></p>
      <p>• <strong>Primary bottleneck</strong>: {primary_bn.upper()} saturates first</p>
      <p>• Tick degradation starts around <strong>{tick_sat} clients</strong></p>
      {"<p>• YOLO is GPU-bound — batching is efficient, consider higher concurrency</p>" if primary_bn not in ("yolo","tick") else ""}
      {"<p>• MediaPipe is CPU-bound — increase MEDIAPIPE_STRIDE or reduce FaceMesh resolution</p>" if primary_bn == "mediapipe" else ""}
    </div>
  </div>
</div>

<script>
const lineOpts = (yLabel, sugMax) => ({{
  responsive:true,
  plugins:{{legend:{{labels:{{color:'#94a3b8'}}}}}},
  scales:{{
    x:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e293b'}},
       title:{{display:true,text:'Concurrent Clients / Instance',color:'#64748b'}}}},
    y:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e293b'}},
       title:{{display:true,text:yLabel,color:'#64748b'}},
       ...(sugMax?{{suggestedMax:sugMax}}:{{}})}}
  }}
}});

new Chart(document.getElementById('tickChart'), {{
  type:'line', data:{{labels:{js(labels)}, datasets:[
    {{label:'Tick Avg (Inst1)', data:{js(tick_avg)}, borderColor:'#60a5fa', backgroundColor:'#60a5fa20', fill:true, tension:0.3}},
    {{label:'Tick P95 (Inst1)', data:{js(tick_p95)}, borderColor:'#f87171', borderDash:[5,5], tension:0.3}},
    {{label:'Tick P99 (Inst1)', data:{js(tick_p99)}, borderColor:'#ef4444', borderDash:[2,4], tension:0.3}},
    {{label:'100ms budget',     data:{js([100]*len(labels))}, borderColor:'#ef444440', borderDash:[3,3], pointRadius:0}}
    {inst2_tick_ds}
  ]}}, options:lineOpts('Latency (ms)', 200)
}});

new Chart(document.getElementById('yoloChart'), {{
  type:'line', data:{{labels:{js(labels)}, datasets:[
    {{label:'YOLO Avg (Inst1)', data:{js(yolo_avg)}, borderColor:'#a78bfa', backgroundColor:'#a78bfa20', fill:true, tension:0.3}},
    {{label:'YOLO P95 (Inst1)', data:{js(yolo_p95)}, borderColor:'#f59e0b', borderDash:[5,5], tension:0.3}},
    {{label:'YOLO P99 (Inst1)', data:{js(yolo_p99)}, borderColor:'#ef4444', borderDash:[2,4], tension:0.3}}
    {inst2_yolo_ds}
  ]}}, options:lineOpts('Latency (ms)')
}});

new Chart(document.getElementById('cpuChart'), {{
  type:'line', data:{{labels:{js(labels)}, datasets:[
    {{label:'CPU Avg % (Inst1)', data:{js(cpu_avg)}, borderColor:'#34d399', backgroundColor:'#34d39920', fill:true, tension:0.3}},
    {{label:'CPU Max % (Inst1)', data:{js(cpu_max)}, borderColor:'#f87171', borderDash:[5,5], tension:0.3}},
    {{label:'85% threshold',     data:{js([85]*len(labels))}, borderColor:'#ef444440', borderDash:[3,3], pointRadius:0}}
    {inst2_cpu_ds}
  ]}}, options:lineOpts('CPU (%)', 100)
}});

new Chart(document.getElementById('hzChart'), {{
  type:'bar', data:{{labels:{js(labels)}, datasets:[
    {{label:'Tick OK %', data:{js(tick_ok)},
      backgroundColor:{js(tick_ok)}.map(v=>v>=95?'#22c55e':v>=80?'#f59e0b':'#ef4444')}}
  ]}},
  options:{{...lineOpts('% of ticks within budget', 100), plugins:{{legend:{{labels:{{color:'#94a3b8'}}}}}}}}
}});

{gpu_chart_js}

{"" if not has_mp else f"""
new Chart(document.getElementById('mpChart'), {{
  type:'line', data:{{labels:{js(labels)}, datasets:[
    {{label:'MP Avg (Inst1)', data:{js(mp_avg)}, borderColor:'#fb923c', backgroundColor:'#fb923c20', fill:true, tension:0.3}},
    {{label:'MP P95 (Inst1)', data:{js(mp_p95)}, borderColor:'#f97316', borderDash:[5,5], tension:0.3}}
    {inst2_mp_ds}
  ]}}, options:lineOpts('MediaPipe Latency (ms)')
}});
"""}

new Chart(document.getElementById('breakdownChart'), {{
  type:'bar', data:{{labels:{js(labels)}, datasets:[
    {{label:'YOLO (ms)',      data:{js(yolo_avg)}, backgroundColor:'#8b5cf6aa', stack:'s'}},
    {{label:'MediaPipe (ms)', data:{js(mp_avg)},   backgroundColor:'#f97316aa', stack:'s'}},
    {{label:'Overhead (ms)',  data:{js(overhead)},  backgroundColor:'#64748baa', stack:'s'}}
  ]}},
  options:{{
    responsive:true,
    plugins:{{legend:{{labels:{{color:'#94a3b8'}}}},
             tooltip:{{callbacks:{{footer: items => 'Note: YOLO+MP run concurrently — stacked here for breakdown only'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e293b'}},stacked:true,
         title:{{display:true,text:'Concurrent Clients / Instance',color:'#64748b'}}}},
      y:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e293b'}},stacked:true,
         title:{{display:true,text:'ms (stacked for breakdown only)',color:'#64748b'}}}}
    }}
  }}
}});

{fps_chart_js}
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n  HTML report → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(
    backend_url  : str,
    backend_url2 : str | None,
    video_path   : str,
    levels       : list[int],
    warmup_s     : int,
    steady_s     : int,
    output_html  : str,
    output_json  : str,
):
    # Suppress aioice STUN timer noise — see _aioice_exception_handler above
    asyncio.get_event_loop().set_exception_handler(_aioice_exception_handler)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE

    print()
    print("=" * 62)
    print("  AI Proctor — Capacity & Performance Benchmark")
    print("=" * 62)
    print(f"  Instance 1 : {backend_url}")
    if backend_url2:
        print(f"  Instance 2 : {backend_url2}")
    print(f"  Video      : {video_path}")
    print(f"  Levels     : {levels} clients/instance")
    print(f"  Per level  : {warmup_s}s warmup + {steady_s}s steady state")
    print("=" * 62)

    print("\n  Fetching system info...", end="", flush=True)
    system_info = await fetch_system_report(backend_url, ssl_ctx)
    print(" done")

    level_results = []
    for n in levels:
        try:
            result = await run_level(
                backend_url  = backend_url,
                video_path   = video_path,
                n_clients    = n,
                warmup_s     = warmup_s,
                steady_s     = steady_s,
                ssl_ctx      = ssl_ctx,
                backend_url2 = backend_url2,
            )
        except Exception as e:
            print(f"\n  ✗ Level {n} crashed ({type(e).__name__}: {e}) — saving partial data and continuing")
            result = {"n_clients": n, "connected": 0, "error": str(e)}
        level_results.append(result)
        await asyncio.sleep(5)

    # Fetch /analysis AFTER all levels — server has accumulated fps_log for the full run
    print("\n  Fetching FPS analysis...", end="", flush=True)
    if backend_url2:
        analysis1, analysis2 = await asyncio.gather(
            fetch_analysis(backend_url,  ssl_ctx),
            fetch_analysis(backend_url2, ssl_ctx),
        )
    else:
        analysis1 = await fetch_analysis(backend_url, ssl_ctx)
        analysis2 = {}
    print(" done")

    print("\n" + "=" * 62)
    print("  BENCHMARK COMPLETE")
    print("=" * 62)

    bottleneck = analyse_bottleneck(level_results)
    print(f"\n  Primary bottleneck  : {bottleneck['bottleneck'].upper()}")
    print(f"  Max safe concurrency: {bottleneck['max_safe']} clients/instance")

    config = {
        "backend_url" : backend_url,
        "backend_url2": backend_url2 or "",
        "video_path"  : video_path,
        "levels"      : levels,
        "warmup_s"    : warmup_s,
        "steady_s"    : steady_s,
        "timestamp"   : datetime.now().isoformat(),
    }

    # Sanitise snapshots from JSON (large arrays → omit raw_log from analysis)
    analysis1_clean = {"by_concurrent": analysis1.get("by_concurrent", {})}
    analysis2_clean = {"by_concurrent": analysis2.get("by_concurrent", {})} if analysis2 else {}

    data = {
        "config"    : config,
        "system"    : system_info,
        "levels"    : level_results,
        "bottleneck": bottleneck,
        "fps_analysis": {"instance1": analysis1_clean, "instance2": analysis2_clean},
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  JSON data   → {output_json}")

    generate_html_report(
        levels      = level_results,
        system_info = system_info,
        bottleneck  = bottleneck,
        config      = config,
        analysis1   = analysis1_clean,
        analysis2   = analysis2_clean,
        output_path = output_html,
    )
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AI Proctor capacity benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single instance
  python benchmark.py --url http://13.x.x.x:8000 --video test_video.mp4 --levels 1,2,4

  # Dual instance (clients connect to both simultaneously)
  python benchmark.py --url http://13.x.x.x:8000 --url2 http://13.x.x.x:8001 \\
      --video test_video.mp4 --levels 1,2,4 --warmup 30 --steady 120
""",
    )
    ap.add_argument("--url",    required=True,  help="Instance 1 backend URL")
    ap.add_argument("--url2",   default="",     help="Instance 2 backend URL (optional)")
    ap.add_argument("--video",  required=True,  help="Path to test .mp4 video")
    ap.add_argument("--levels", default="1,2,4",
                    help="Comma-separated concurrency levels per instance (default: 1,2,4)")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warmup seconds per level (default: 20)")
    ap.add_argument("--steady", type=int, default=60,
                    help="Steady-state measurement seconds per level (default: 60)")
    ap.add_argument("--output", default="",     help="Output base name (auto if omitted)")
    args = ap.parse_args()

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base   = args.output or f"benchmark_{ts}"
    levels = [int(x.strip()) for x in args.levels.split(",")]

    asyncio.run(main(
        backend_url  = args.url,
        backend_url2 = args.url2 or None,
        video_path   = args.video,
        levels       = levels,
        warmup_s     = args.warmup,
        steady_s     = args.steady,
        output_html  = f"{base}.html",
        output_json  = f"{base}.json",
    ))
