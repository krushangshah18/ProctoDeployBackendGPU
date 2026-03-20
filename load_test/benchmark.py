"""
AI Proctor — Capacity & Performance Benchmark

Runs the load test at multiple concurrency levels (1, 5, 10, 25, 50) and
generates a comprehensive HTML report showing:
  - Resource utilisation (CPU, RAM, GPU, VRAM) vs concurrency
  - YOLO and tick latency vs concurrency
  - FPS-per-user estimates
  - Bottleneck identification (which resource saturates first)
  - Maximum safe concurrency recommendation

Usage:
    python benchmark.py --url http://localhost:8000 --video test_video.mp4
    python benchmark.py --url http://<ec2-ip>:8000  --video test_video.mp4 --levels 1,5,10,25,50
    python benchmark.py --url http://<ec2-ip>:8000  --video test_video.mp4 --levels 1,5 --warmup 30 --steady 120

Output:
    benchmark_report_<timestamp>.html   (open in browser)
    benchmark_data_<timestamp>.json     (raw data)
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


# ── Metrics collection ────────────────────────────────────────────────────────

async def collect_metrics(backend_url: str, duration_s: float, ssl_ctx, interval: float = 3.0) -> list[dict]:
    """Poll /metrics every `interval` seconds for `duration_s` seconds."""
    snapshots = []
    deadline  = time.time() + duration_s
    conn      = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(connector=conn) as http:
        while time.time() < deadline:
            try:
                async with http.get(f"{backend_url}/metrics",
                                    timeout=aiohttp.ClientTimeout(total=5)) as resp:
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
                            "yolo_avg_ms"      : yolo.get("lat_avg_ms",        0.0),
                            "yolo_p95_ms"      : yolo.get("lat_p95_ms",        0.0),
                            "yolo_p99_ms"      : yolo.get("lat_p99_ms",        0.0),
                            "cpu_pct"          : sys_.get("cpu_percent",       0.0),
                            "ram_mb"           : sys_.get("mem_rss_mb",        0.0),
                            "gpu_util"         : sys_.get("gpu_util_pct",      None),
                            "vram_mb"          : sys_.get("gpu_mem_used_mb",   None),
                            "mediapipe_avg_ms" : mp_.get("lat_avg_ms",         0.0),
                            "mediapipe_p95_ms" : mp_.get("lat_p95_ms",         0.0),
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
            async with http.get(f"{backend_url}/system/report",
                                timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


# ── Single level test ─────────────────────────────────────────────────────────

async def run_level(
    backend_url : str,
    video_path  : str,
    n_clients   : int,
    warmup_s    : int,
    steady_s    : int,
    ssl_ctx,
) -> dict:
    """
    Run one concurrency level:
      1. Connect all clients
      2. Warmup period (skip metrics — system is stabilising)
      3. Steady-state metrics collection
      4. Disconnect all clients
    """
    print(f"\n  {'─'*54}")
    print(f"  Level: {n_clients} concurrent client(s)")
    print(f"  {'─'*54}")

    results: dict = {}

    # Connect all clients simultaneously
    print(f"  Connecting {n_clients} client(s)...", end="", flush=True)
    t0     = time.perf_counter()
    tasks  = [
        asyncio.create_task(
            run_candidate(
                backend_url  = backend_url,
                video_path   = video_path,
                candidate_id = f"bench_{n_clients:03d}_{i+1:03d}",
                duration_s   = warmup_s + steady_s + 10,
                results      = results,
                ssl_verify   = False,
            )
        )
        for i in range(n_clients)
    ]
    # Give clients a moment to establish connections
    await asyncio.sleep(min(warmup_s, 15))
    connect_ms_elapsed = (time.perf_counter() - t0) * 1000
    connected = sum(1 for r in results.values() if r.get("status") in ("connected", "completed"))
    print(f" {connected}/{n_clients} connected in {connect_ms_elapsed:.0f}ms")

    if connected == 0:
        print("  ✗ No clients connected — skipping level")
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"n_clients": n_clients, "connected": 0, "error": "no_clients_connected"}

    # Warmup (let system stabilise)
    if warmup_s > 0:
        print(f"  Warmup {warmup_s}s...", end="", flush=True)
        await asyncio.sleep(warmup_s)
        print(" done")

    # Steady-state measurement
    print(f"  Measuring {steady_s}s steady state...", end="", flush=True)
    snapshots = await collect_metrics(backend_url, steady_s, ssl_ctx, interval=2.0)
    print(f" {len(snapshots)} samples collected")

    # Disconnect clients
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(3)   # let server clean up sessions

    # Aggregate metrics
    def stats(vals):
        if not vals: return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "min": 0}
        s = sorted(vals)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 2),
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[int(n * 0.95)], 2),
            "p99": round(s[min(int(n * 0.99), n-1)], 2),
            "max": round(max(s), 2),
            "min": round(min(s), 2),
        }

    tick_vals  = [s["tick_ms"]          for s in snapshots if s["tick_ms"]          > 0]
    yolo_vals  = [s["yolo_avg_ms"]      for s in snapshots if s["yolo_avg_ms"]      > 0]
    yolo_p95s  = [s["yolo_p95_ms"]      for s in snapshots if s["yolo_p95_ms"]      > 0]
    cpu_vals   = [s["cpu_pct"]          for s in snapshots if s["cpu_pct"]          > 0]
    ram_vals   = [s["ram_mb"]           for s in snapshots]
    gpu_vals   = [s["gpu_util"]         for s in snapshots if s.get("gpu_util")     is not None]
    vram_vals  = [s["vram_mb"]          for s in snapshots if s.get("vram_mb")      is not None]
    mp_vals    = [s["mediapipe_avg_ms"] for s in snapshots if s["mediapipe_avg_ms"] > 0]
    audio_vals = [s["audio_avg_ms"]     for s in snapshots if s["audio_avg_ms"]     > 0]

    connect_times = [r["connect_ms"] for r in results.values() if "connect_ms" in r]

    # FPS per user = 10Hz target / sessions (rough estimate from tick)
    # More precise: if tick < 100ms the full 10Hz is maintained per user
    ticks_ok    = sum(1 for v in tick_vals if v <= 100)
    tick_ok_pct = 100 * ticks_ok / len(tick_vals) if tick_vals else 0

    # Effective Hz = 1000 / avg_tick_ms
    eff_hz = 1000 / stats(tick_vals)["avg"] if tick_vals and stats(tick_vals)["avg"] > 0 else 0

    level_data = {
        "n_clients"       : n_clients,
        "connected"       : connected,
        "connect_ms"      : stats(connect_times),
        "tick_ms"         : stats(tick_vals),
        "yolo_avg_ms"     : stats(yolo_vals),
        "yolo_p95_ms"     : stats(yolo_p95s),
        "cpu_pct"         : stats(cpu_vals),
        "ram_mb"          : stats(ram_vals),
        "gpu_util_pct"    : stats(gpu_vals)   if gpu_vals   else None,
        "vram_mb"         : stats(vram_vals)  if vram_vals  else None,
        "mediapipe_avg_ms": stats(mp_vals)    if mp_vals    else None,
        "audio_avg_ms"    : stats(audio_vals) if audio_vals else None,
        "tick_ok_pct"     : round(tick_ok_pct, 1),
        "effective_hz"    : round(eff_hz, 2),
        "samples"         : len(snapshots),
        "snapshots"       : snapshots,
    }

    _print_level_summary(level_data)
    return level_data


def _print_level_summary(d: dict):
    n   = d["n_clients"]
    ok  = d["tick_ok_pct"]
    hz  = d["effective_hz"]
    verdict = "✓ PASS" if ok >= 95 else ("⚠ WARN" if ok >= 80 else "✗ FAIL")

    print(f"\n  Results for {n} client(s):")
    print(f"    Tick:  avg={d['tick_ms']['avg']:.1f}ms  p95={d['tick_ms']['p95']:.1f}ms  "
          f"10Hz maintained={ok:.1f}%  eff={hz:.1f}Hz  → {verdict}")
    print(f"    YOLO:  avg={d['yolo_avg_ms']['avg']:.1f}ms  p95={d['yolo_avg_ms']['p95']:.1f}ms")
    print(f"    CPU:   avg={d['cpu_pct']['avg']:.1f}%  max={d['cpu_pct']['max']:.1f}%")
    print(f"    RAM:   avg={d['ram_mb']['avg']:.0f}MB  max={d['ram_mb']['max']:.0f}MB")
    if d.get("gpu_util_pct"):
        print(f"    GPU:   avg={d['gpu_util_pct']['avg']:.1f}%  max={d['gpu_util_pct']['max']:.1f}%")
    if d.get("vram_mb"):
        print(f"    VRAM:  avg={d['vram_mb']['avg']:.0f}MB  max={d['vram_mb']['max']:.0f}MB")
    if d.get("mediapipe_avg_ms"):
        print(f"    MP:    avg={d['mediapipe_avg_ms']['avg']:.1f}ms  p95={d['mediapipe_avg_ms']['p95']:.1f}ms")
    if d.get("audio_avg_ms"):
        print(f"    Audio: avg={d['audio_avg_ms']['avg']:.1f}ms  p95={d['audio_avg_ms']['p95']:.1f}ms")


# ── Bottleneck analysis ───────────────────────────────────────────────────────

def analyse_bottleneck(levels: list[dict]) -> dict:
    """
    Determine which resource saturates first as concurrency increases.
    Returns bottleneck name, max safe concurrency, and per-resource analysis.
    """
    valid = [l for l in levels if l.get("connected", 0) > 0 and "tick_ms" in l]
    if not valid:
        return {"bottleneck": "unknown", "max_safe": 0, "analysis": {}}

    analysis = {}

    # CPU saturation threshold: > 85% average
    cpu_saturate = next((l["n_clients"] for l in valid if l["cpu_pct"]["avg"] > 85), None)
    analysis["cpu"] = {
        "values"    : {l["n_clients"]: l["cpu_pct"]["avg"] for l in valid},
        "saturates_at": cpu_saturate,
        "threshold" : 85,
        "unit"      : "%",
    }

    # Tick degradation: < 95% of ticks within 100ms
    tick_degrade = next((l["n_clients"] for l in valid if l["tick_ok_pct"] < 95), None)
    analysis["tick"] = {
        "values"      : {l["n_clients"]: l["tick_ok_pct"] for l in valid},
        "saturates_at": tick_degrade,
        "threshold"   : 95,
        "unit"        : "% ok",
    }

    # YOLO latency: p95 > 200ms means batching overhead
    yolo_degrade = next((l["n_clients"] for l in valid if l["yolo_avg_ms"]["p95"] > 200), None)
    analysis["yolo"] = {
        "values"      : {l["n_clients"]: l["yolo_avg_ms"]["avg"] for l in valid},
        "saturates_at": yolo_degrade,
        "threshold"   : 200,
        "unit"        : "ms",
    }

    # RAM: > 80% of system RAM (estimated at 8GB if unknown)
    ram_vals = {l["n_clients"]: l["ram_mb"]["avg"] for l in valid}
    analysis["ram"] = {
        "values"      : ram_vals,
        "saturates_at": None,
        "unit"        : "MB",
    }

    # GPU (if available)
    gpu_levels = [l for l in valid if l.get("gpu_util_pct")]
    if gpu_levels:
        gpu_saturate = next((l["n_clients"] for l in gpu_levels if l["gpu_util_pct"]["avg"] > 85), None)
        analysis["gpu"] = {
            "values"      : {l["n_clients"]: l["gpu_util_pct"]["avg"] for l in gpu_levels},
            "saturates_at": gpu_saturate,
            "threshold"   : 85,
            "unit"        : "%",
        }

    # Max safe concurrency = largest level where tick is still OK
    max_safe = max(
        (l["n_clients"] for l in valid if l["tick_ok_pct"] >= 95),
        default=0
    )

    # Primary bottleneck = first resource to saturate
    candidates = {
        name: info["saturates_at"]
        for name, info in analysis.items()
        if info.get("saturates_at") is not None
    }
    bottleneck = min(candidates, key=lambda k: candidates[k]) if candidates else "none_detected"

    return {
        "bottleneck"    : bottleneck,
        "max_safe"      : max_safe,
        "analysis"      : analysis,
    }


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_html_report(
    levels      : list[dict],
    system_info : dict,
    bottleneck  : dict,
    config      : dict,
    output_path : str,
):
    labels        = [str(l["n_clients"]) for l in levels if l.get("connected", 0) > 0]
    valid         = [l for l in levels if l.get("connected", 0) > 0]
    tick_avg      = [l["tick_ms"]["avg"]        for l in valid]
    tick_p95      = [l["tick_ms"]["p95"]        for l in valid]
    yolo_avg      = [l["yolo_avg_ms"]["avg"]    for l in valid]
    yolo_p95      = [l["yolo_avg_ms"]["p95"]    for l in valid]
    cpu_avg       = [l["cpu_pct"]["avg"]        for l in valid]
    cpu_max       = [l["cpu_pct"]["max"]        for l in valid]
    ram_avg       = [l["ram_mb"]["avg"]         for l in valid]
    tick_ok       = [l["tick_ok_pct"]           for l in valid]
    eff_hz        = [l["effective_hz"]          for l in valid]
    has_gpu       = any(l.get("gpu_util_pct") for l in valid)
    gpu_avg       = [l["gpu_util_pct"]["avg"] if l.get("gpu_util_pct") else 0 for l in valid]
    vram_avg      = [l["vram_mb"]["avg"]      if l.get("vram_mb")      else 0 for l in valid]
    has_mp        = any(l.get("mediapipe_avg_ms") for l in valid)
    mp_avg        = [l["mediapipe_avg_ms"]["avg"] if l.get("mediapipe_avg_ms") else 0 for l in valid]
    mp_p95        = [l["mediapipe_avg_ms"]["p95"] if l.get("mediapipe_avg_ms") else 0 for l in valid]
    has_audio     = any(l.get("audio_avg_ms") for l in valid)
    audio_avg     = [l["audio_avg_ms"]["avg"] if l.get("audio_avg_ms") else 0 for l in valid]
    audio_p95     = [l["audio_avg_ms"]["p95"] if l.get("audio_avg_ms") else 0 for l in valid]

    env   = system_info.get("environment", {})
    hw    = system_info.get("hardware",    {})
    det   = system_info.get("detector",    {})

    max_safe    = bottleneck["max_safe"]
    primary_bn  = bottleneck["bottleneck"]
    tick_sat    = bottleneck["analysis"].get("tick", {}).get("saturates_at", "N/A")

    bn_color = {"cpu": "#ef4444", "tick": "#f59e0b", "yolo": "#8b5cf6",
                "gpu": "#06b6d4", "ram": "#10b981", "none_detected": "#22c55e"}
    bn_clr = bn_color.get(primary_bn, "#94a3b8")

    def js_arr(lst): return json.dumps(lst)

    gpu_charts = ""
    if has_gpu:
        gpu_charts = f"""
        <div class="chart-card">
          <h3>GPU Utilisation vs Concurrency</h3>
          <canvas id="gpuChart"></canvas>
        </div>
        <div class="chart-card">
          <h3>VRAM Usage vs Concurrency</h3>
          <canvas id="vramChart"></canvas>
        </div>"""

    mp_audio_chart = ""
    if has_mp or has_audio:
        mp_audio_chart = """
        <div class="chart-card">
          <h3>MediaPipe &amp; Audio VAD Latency vs Concurrency</h3>
          <canvas id="mpAudioChart"></canvas>
        </div>"""

    gpu_js = ""
    if has_gpu:
        gpu_js = f"""
        new Chart(document.getElementById('gpuChart'), {{
          type: 'line', data: {{
            labels: {js_arr(labels)},
            datasets: [{{label:'GPU Avg %', data:{js_arr(gpu_avg)}, borderColor:'#06b6d4', backgroundColor:'#06b6d420', fill:true, tension:0.3}}]
          }}, options: lineOpts('GPU Utilisation (%)', 100)
        }});
        new Chart(document.getElementById('vramChart'), {{
          type: 'line', data: {{
            labels: {js_arr(labels)},
            datasets: [{{label:'VRAM Avg MB', data:{js_arr(vram_avg)}, borderColor:'#8b5cf6', backgroundColor:'#8b5cf620', fill:true, tension:0.3}}]
          }}, options: lineOpts('VRAM (MB)')
        }});"""

    mp_audio_js = ""
    if has_mp or has_audio:
        mp_datasets = []
        if has_mp:
            mp_datasets.append(
                f"{{label:'MediaPipe Avg (ms)', data:{js_arr(mp_avg)}, borderColor:'#fb923c', backgroundColor:'#fb923c20', fill:true, tension:0.3}}"
            )
            mp_datasets.append(
                f"{{label:'MediaPipe P95 (ms)', data:{js_arr(mp_p95)}, borderColor:'#f97316', borderDash:[5,5], tension:0.3}}"
            )
        if has_audio:
            mp_datasets.append(
                f"{{label:'Audio VAD Avg (ms)', data:{js_arr(audio_avg)}, borderColor:'#4ade80', backgroundColor:'#4ade8020', fill:false, tension:0.3}}"
            )
            mp_datasets.append(
                f"{{label:'Audio VAD P95 (ms)', data:{js_arr(audio_p95)}, borderColor:'#22c55e', borderDash:[5,5], tension:0.3}}"
            )
        mp_audio_js = f"""
        new Chart(document.getElementById('mpAudioChart'), {{
          type: 'line', data: {{
            labels: {js_arr(labels)},
            datasets: [{', '.join(mp_datasets)}]
          }}, options: lineOpts('Latency (ms)')
        }});"""

    table_rows = ""
    for l in valid:
        ok  = l["tick_ok_pct"]
        clr = "#22c55e" if ok >= 95 else ("#f59e0b" if ok >= 80 else "#ef4444")
        vrd = "✓" if ok >= 95 else ("⚠" if ok >= 80 else "✗")
        gpu_td  = f"{l['gpu_util_pct']['avg']:.1f}%"    if l.get("gpu_util_pct")    else "—"
        vram_td = f"{l['vram_mb']['avg']:.0f}MB"        if l.get("vram_mb")         else "—"
        mp_td   = f"{l['mediapipe_avg_ms']['avg']:.1f}" if l.get("mediapipe_avg_ms") else "—"
        aud_td  = f"{l['audio_avg_ms']['avg']:.1f}"     if l.get("audio_avg_ms")    else "—"
        table_rows += f"""
        <tr>
          <td>{l['n_clients']}</td>
          <td>{l['connected']}</td>
          <td>{l['tick_ms']['avg']:.1f}</td>
          <td>{l['tick_ms']['p95']:.1f}</td>
          <td>{l['yolo_avg_ms']['avg']:.1f}</td>
          <td>{l['yolo_avg_ms']['p95']:.1f}</td>
          <td>{mp_td}</td>
          <td>{aud_td}</td>
          <td>{l['cpu_pct']['avg']:.1f}%</td>
          <td>{l['ram_mb']['avg']:.0f}</td>
          <td>{gpu_td}</td>
          <td>{vram_td}</td>
          <td>{l['effective_hz']:.1f}</td>
          <td style="color:{clr};font-weight:bold">{vrd} {ok:.0f}%</td>
        </tr>"""

    analysis_rows = ""
    for name, info in bottleneck["analysis"].items():
        sat = info.get("saturates_at")
        sat_str = f"at {sat} clients" if sat else "not reached"
        analysis_rows += f"<tr><td>{name.upper()}</td><td>{sat_str}</td><td>{info.get('threshold','—')}{info.get('unit','')}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Proctor — Performance Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1   {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2   {{ font-size: 1.2rem; color: #94a3b8; margin: 32px 0 16px; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
  h3   {{ font-size: 0.95rem; color: #94a3b8; margin-bottom: 12px; }}
  .subtitle {{ color: #64748b; margin-bottom: 32px; font-size: 0.9rem; }}
  .grid-3  {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
  .grid-2  {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }}
  .card    {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
  .card-highlight {{ background: #1e293b; border-radius: 12px; padding: 20px;
                    border: 2px solid {bn_clr}; }}
  .stat-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; margin: 4px 0; }}
  .stat-sub   {{ font-size: 0.8rem; color: #94a3b8; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }}
  .chart-card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
  table  {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th     {{ background: #0f172a; padding: 10px 12px; text-align: left; color: #64748b;
            font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td     {{ padding: 10px 12px; border-bottom: 1px solid #0f172a; }}
  tr:hover td {{ background: #263548; }}
  .tag   {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; }}
  .tag-green  {{ background: #14532d; color: #4ade80; }}
  .tag-yellow {{ background: #713f12; color: #fbbf24; }}
  .tag-red    {{ background: #7f1d1d; color: #f87171; }}
  .info-grid  {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.85rem; }}
  .info-row   {{ display: flex; justify-content: space-between; padding: 6px 0;
                 border-bottom: 1px solid #0f172a; }}
  .info-key   {{ color: #64748b; }}
  .info-val   {{ color: #e2e8f0; font-weight: 500; }}
</style>
</head>
<body>

<h1>AI Proctor — Capacity Benchmark Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Backend: {config['backend_url']}</p>

<h2>Summary</h2>
<div class="grid-3">
  <div class="card-highlight">
    <div class="stat-label">Max Safe Concurrency</div>
    <div class="stat-value" style="color:{bn_clr}">{max_safe}</div>
    <div class="stat-sub">clients at stable 10 Hz</div>
  </div>
  <div class="card">
    <div class="stat-label">Primary Bottleneck</div>
    <div class="stat-value" style="color:{bn_clr};font-size:1.4rem">{primary_bn.upper()}</div>
    <div class="stat-sub">first resource to saturate</div>
  </div>
  <div class="card">
    <div class="stat-label">Levels Tested</div>
    <div class="stat-value">{len(valid)}</div>
    <div class="stat-sub">concurrency levels</div>
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
      <div class="info-row"><span class="info-key">RAM</span><span class="info-val">{hw.get('ram_total_gb','?')} GB</span></div>
    </div>
  </div>
  <div class="card">
    <h3>Inference Device</h3>
    <div class="info-grid">
      <div class="info-row"><span class="info-key">Device</span><span class="info-val">{det.get('device','?')}</span></div>
      <div class="info-row"><span class="info-key">FP16</span><span class="info-val">{det.get('half_precision','?')}</span></div>
      <div class="info-row"><span class="info-key">GPU</span><span class="info-val">{det.get('gpu_name','CPU')}</span></div>
      <div class="info-row"><span class="info-key">VRAM</span><span class="info-val">{det.get('gpu_vram_gb','—')} GB</span></div>
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
    <h3>10 Hz Maintenance vs Concurrency</h3>
    <canvas id="hzChart"></canvas>
  </div>
  {gpu_charts}
  {mp_audio_chart}
</div>

<h2>Detailed Results Table</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Clients</th><th>Connected</th>
        <th>Tick Avg(ms)</th><th>Tick P95(ms)</th>
        <th>YOLO Avg(ms)</th><th>YOLO P95(ms)</th>
        <th>MP Avg(ms)</th><th>VAD Avg(ms)</th>
        <th>CPU Avg</th><th>RAM(MB)</th>
        <th>GPU</th><th>VRAM</th>
        <th>Eff. Hz</th><th>10Hz OK</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

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
    <div style="font-size:0.85rem;line-height:1.8">
      <p>• <strong>Max safe concurrency</strong>: <span style="color:{bn_clr}">{max_safe} clients</span></p>
      <p>• <strong>Primary bottleneck</strong>: {primary_bn.upper()} saturates first</p>
      <p>• Consider scaling horizontally if {primary_bn.upper()} is the limit</p>
      <p>• Tick degradation begins around {tick_sat} clients</p>
    </div>
  </div>
</div>

<script>
const lineOpts = (yLabel, sugMax) => ({{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }}, title: {{ display:true, text:'Concurrent Clients', color:'#64748b' }} }},
    y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }},
         title: {{ display:true, text:yLabel, color:'#64748b' }},
         ...(sugMax ? {{suggestedMax: sugMax}} : {{}}) }}
  }}
}});

new Chart(document.getElementById('tickChart'), {{
  type: 'line',
  data: {{
    labels: {js_arr(labels)},
    datasets: [
      {{label:'Tick Avg (ms)', data:{js_arr(tick_avg)}, borderColor:'#60a5fa', backgroundColor:'#60a5fa20', fill:true, tension:0.3}},
      {{label:'Tick P95 (ms)', data:{js_arr(tick_p95)}, borderColor:'#f87171', borderDash:[5,5], tension:0.3}},
      {{label:'100ms limit',   data:{js_arr([100]*len(labels))}, borderColor:'#ef444466', borderDash:[3,3], pointRadius:0}}
    ]
  }}, options: lineOpts('Latency (ms)', 200)
}});

new Chart(document.getElementById('yoloChart'), {{
  type: 'line',
  data: {{
    labels: {js_arr(labels)},
    datasets: [
      {{label:'YOLO Avg (ms)', data:{js_arr(yolo_avg)}, borderColor:'#a78bfa', backgroundColor:'#a78bfa20', fill:true, tension:0.3}},
      {{label:'YOLO P95 (ms)', data:{js_arr(yolo_p95)}, borderColor:'#f59e0b', borderDash:[5,5], tension:0.3}}
    ]
  }}, options: lineOpts('Latency (ms)')
}});

new Chart(document.getElementById('cpuChart'), {{
  type: 'line',
  data: {{
    labels: {js_arr(labels)},
    datasets: [
      {{label:'CPU Avg %', data:{js_arr(cpu_avg)}, borderColor:'#34d399', backgroundColor:'#34d39920', fill:true, tension:0.3}},
      {{label:'CPU Max %', data:{js_arr(cpu_max)}, borderColor:'#f87171', borderDash:[5,5], tension:0.3}},
      {{label:'85% threshold', data:{js_arr([85]*len(labels))}, borderColor:'#ef444466', borderDash:[3,3], pointRadius:0}}
    ]
  }}, options: lineOpts('CPU (%)', 100)
}});

new Chart(document.getElementById('hzChart'), {{
  type: 'bar',
  data: {{
    labels: {js_arr(labels)},
    datasets: [
      {{label:'10Hz maintained %', data:{js_arr(tick_ok)},
        backgroundColor: {js_arr(tick_ok)}.map(v => v>=95?'#22c55e':v>=80?'#f59e0b':'#ef4444')}}
    ]
  }},
  options: {{
    ...lineOpts('% of ticks within 100ms', 100),
    plugins: {{ legend: {{ labels: {{ color:'#94a3b8' }} }} }}
  }}
}});

{gpu_js}
{mp_audio_js}
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n  HTML report → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    backend_url : str,
    video_path  : str,
    levels      : list[int],
    warmup_s    : int,
    steady_s    : int,
    output_html : str,
    output_json : str,
):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE

    print()
    print("=" * 58)
    print("  AI Proctor — Capacity & Performance Benchmark")
    print("=" * 58)
    print(f"  Backend  : {backend_url}")
    print(f"  Video    : {video_path}")
    print(f"  Levels   : {levels}")
    print(f"  Per level: {warmup_s}s warmup + {steady_s}s steady state")
    print("=" * 58)

    # Fetch system info once at start
    print("\n  Fetching system info...", end="", flush=True)
    system_info = await fetch_system_report(backend_url, ssl_ctx)
    print(" done")

    level_results = []
    for n in levels:
        result = await run_level(
            backend_url = backend_url,
            video_path  = video_path,
            n_clients   = n,
            warmup_s    = warmup_s,
            steady_s    = steady_s,
            ssl_ctx     = ssl_ctx,
        )
        level_results.append(result)
        await asyncio.sleep(5)   # cooldown between levels

    print("\n" + "=" * 58)
    print("  BENCHMARK COMPLETE")
    print("=" * 58)

    # Analyse bottlenecks
    bottleneck = analyse_bottleneck(level_results)
    print(f"\n  Primary bottleneck : {bottleneck['bottleneck'].upper()}")
    print(f"  Max safe concurrency: {bottleneck['max_safe']} clients")

    config = {
        "backend_url": backend_url,
        "video_path" : video_path,
        "levels"     : levels,
        "warmup_s"   : warmup_s,
        "steady_s"   : steady_s,
        "timestamp"  : datetime.now().isoformat(),
    }

    # Save JSON
    data = {"config": config, "system": system_info,
            "levels": level_results, "bottleneck": bottleneck}
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  JSON data   → {output_json}")

    # Generate HTML report
    generate_html_report(level_results, system_info, bottleneck, config, output_html)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AI Proctor capacity benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick local test
  python benchmark.py --url http://localhost:8000 --video test_video.mp4 --levels 1,3,5

  # Full EC2 benchmark
  python benchmark.py --url http://54.x.x.x:8000 --video test_video.mp4 --levels 1,5,10,25,50 --warmup 30 --steady 120
""",
    )
    ap.add_argument("--url",      required=True,  help="Backend URL")
    ap.add_argument("--video",    required=True,  help="Path to test .mp4 video")
    ap.add_argument("--levels",   default="1,5,10,25,50",
                    help="Comma-separated concurrency levels (default: 1,5,10,25,50)")
    ap.add_argument("--warmup",   type=int, default=20,
                    help="Warmup seconds per level before measuring (default: 20)")
    ap.add_argument("--steady",   type=int, default=60,
                    help="Steady-state measurement seconds per level (default: 60)")
    ap.add_argument("--output",   default="",     help="Output base name (auto if omitted)")
    args = ap.parse_args()

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base   = args.output or f"benchmark_{ts}"
    levels = [int(x.strip()) for x in args.levels.split(",")]

    asyncio.run(main(
        backend_url = args.url,
        video_path  = args.video,
        levels      = levels,
        warmup_s    = args.warmup,
        steady_s    = args.steady,
        output_html = f"{base}.html",
        output_json = f"{base}.json",
    ))
