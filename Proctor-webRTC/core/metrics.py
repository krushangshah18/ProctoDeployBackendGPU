"""
MetricsCollector — lightweight in-process metrics for production monitoring.

Tracks:
  • HTTP request counts, error rates, and latency (p50 / p95 / p99) per endpoint
  • WebRTC session counts (total created, currently active)
  • Alert / warning totals
  • System resources: CPU %, RSS MB, GPU util %, VRAM MB  (sampled every 5 s)
  • YOLO batch inference latency (rolling 100 samples)
  • Coordinator tick latency    (rolling 100 samples)

Usage:
    from core.metrics import metrics          # module-level singleton

    metrics.record_request("/snapshot/x", 200, 12.4)
    metrics.inc_session()
    metrics.inc_alert()
    snap = metrics.snapshot()                 # → dict (used by /metrics endpoint)
"""
from __future__ import annotations

import math
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import psutil
import torch

_PROC = psutil.Process()


class MetricsCollector:

    def __init__(self, resource_sample_interval_s: float = 5.0) -> None:
        self._lock       = threading.Lock()
        self._start_time = time.time()

        # ── HTTP requests ─────────────────────────────────────────────────────
        self._req_count:         dict[str, int]   = defaultdict(int)
        self._req_errors:        dict[str, int]   = defaultdict(int)
        self._req_latency:       dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._req_total:         int              = 0
        self._req_errors_total:  int              = 0

        # ── Sessions ──────────────────────────────────────────────────────────
        self._sessions_created:  int = 0
        self._sessions_active:   int = 0

        # ── Events ────────────────────────────────────────────────────────────
        self._alerts_total:      int = 0
        self._warnings_total:    int = 0

        # ── Inference / tick timing ───────────────────────────────────────────
        self._yolo_latencies:    deque = deque(maxlen=200)
        self._tick_latencies:    deque = deque(maxlen=200)

        # ── Resource snapshots (updated by background thread) ─────────────────
        self._cpu_percent:       float = 0.0
        self._mem_rss_mb:        float = 0.0
        self._gpu_util_pct:      float = 0.0
        self._gpu_mem_used_mb:   float = 0.0
        self._gpu_mem_total_mb:  float = 0.0

        # ── Background sampler ────────────────────────────────────────────────
        self._stop_event     = threading.Event()
        self._sampler        = threading.Thread(
            target=self._resource_sampler,
            args=(resource_sample_interval_s,),
            daemon=True,
            name="metrics-sampler",
        )
        self._sampler.start()

    # ── Recording API (called from hot paths — must be fast) ──────────────────

    def record_request(self, endpoint: str, status_code: int,
                       latency_ms: float) -> None:
        with self._lock:
            self._req_count[endpoint]   += 1
            self._req_total             += 1
            self._req_latency[endpoint].append(latency_ms)
            if status_code >= 400:
                self._req_errors[endpoint] += 1
                self._req_errors_total     += 1

    def inc_session(self) -> None:
        with self._lock:
            self._sessions_created += 1
            self._sessions_active  += 1

    def dec_session(self) -> None:
        with self._lock:
            self._sessions_active = max(0, self._sessions_active - 1)

    def inc_alert(self) -> None:
        with self._lock:
            self._alerts_total += 1

    def inc_warning(self) -> None:
        with self._lock:
            self._warnings_total += 1

    def record_yolo_latency(self, ms: float) -> None:
        with self._lock:
            self._yolo_latencies.append(ms)

    def record_tick_latency(self, ms: float) -> None:
        with self._lock:
            self._tick_latencies.append(ms)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a complete metrics dict safe for JSON serialisation."""
        with self._lock:
            uptime_s = time.time() - self._start_time

            # Per-endpoint aggregates
            endpoints: dict[str, dict] = {}
            for ep, count in sorted(self._req_count.items()):
                lats = sorted(self._req_latency[ep])
                n    = len(lats)
                endpoints[ep] = {
                    "count"      : count,
                    "errors"     : self._req_errors.get(ep, 0),
                    "lat_avg_ms" : round(_mean(lats), 1),
                    "lat_p50_ms" : round(_pct(lats, 50), 1),
                    "lat_p95_ms" : round(_pct(lats, 95), 1),
                    "lat_p99_ms" : round(_pct(lats, 99), 1),
                }

            yolo = sorted(self._yolo_latencies)
            tick = sorted(self._tick_latencies)

            gpu_pct = 0.0
            if self._gpu_mem_total_mb > 0:
                gpu_pct = round(
                    100 * self._gpu_mem_used_mb / self._gpu_mem_total_mb, 1
                )

            return {
                "generated_at" : datetime.now(timezone.utc).isoformat(),
                "uptime_s"     : round(uptime_s, 1),
                "uptime"       : _fmt_uptime(uptime_s),

                "requests": {
                    "total"          : self._req_total,
                    "errors"         : self._req_errors_total,
                    "error_rate_pct" : round(
                        100 * self._req_errors_total / max(1, self._req_total), 2
                    ),
                    "by_endpoint"    : endpoints,
                },

                "sessions": {
                    "total_created": self._sessions_created,
                    "active"       : self._sessions_active,
                },

                "events": {
                    "alerts_total"  : self._alerts_total,
                    "warnings_total": self._warnings_total,
                },

                "yolo": {
                    "samples"       : len(yolo),
                    "lat_avg_ms"    : round(_mean(yolo), 1),
                    "lat_p95_ms"    : round(_pct(yolo, 95), 1),
                    "lat_p99_ms"    : round(_pct(yolo, 99), 1),
                    "lat_max_ms"    : round(max(yolo), 1) if yolo else 0.0,
                },

                "coordinator": {
                    "tick_avg_ms"   : round(_mean(tick), 1),
                    "tick_p95_ms"   : round(_pct(tick, 95), 1),
                    "tick_max_ms"   : round(max(tick), 1) if tick else 0.0,
                },

                "system": {
                    "cpu_percent"       : self._cpu_percent,
                    "mem_rss_mb"        : round(self._mem_rss_mb, 1),
                    "gpu_util_pct"      : self._gpu_util_pct,
                    "gpu_mem_used_mb"   : round(self._gpu_mem_used_mb, 1),
                    "gpu_mem_total_mb"  : round(self._gpu_mem_total_mb, 1),
                    "gpu_mem_used_pct"  : gpu_pct,
                },
            }

    def stop(self) -> None:
        self._stop_event.set()

    # ── Background resource sampler ───────────────────────────────────────────

    def _resource_sampler(self, interval: float) -> None:
        # Prime psutil — first call always returns 0.0
        psutil.cpu_percent(interval=None)

        while not self._stop_event.wait(interval):
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = _PROC.memory_info().rss / 1024 / 1024

                gpu_util  = 0.0
                gpu_used  = 0.0
                gpu_total = 0.0

                if torch.cuda.is_available():
                    gpu_used  = torch.cuda.memory_allocated(0) / 1024 / 1024
                    gpu_total = (
                        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    )
                    # Try pynvml for hardware utilization (optional dep)
                    try:
                        import pynvml  # type: ignore
                        pynvml.nvmlInit()
                        h        = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util_obj = pynvml.nvmlDeviceGetUtilizationRates(h)
                        gpu_util = float(util_obj.gpu)
                    except Exception:
                        gpu_util = 0.0

                with self._lock:
                    self._cpu_percent      = cpu
                    self._mem_rss_mb       = mem
                    self._gpu_util_pct     = gpu_util
                    self._gpu_mem_used_mb  = gpu_used
                    self._gpu_mem_total_mb = gpu_total

            except Exception:
                pass


# ── Small math helpers ────────────────────────────────────────────────────────

def _pct(sorted_data: list, p: float) -> float:
    if not sorted_data:
        return 0.0
    k  = (len(sorted_data) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_data[lo])
    return float(sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo))


def _mean(data: list | deque) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)


def _fmt_uptime(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# ── Module-level singleton ────────────────────────────────────────────────────
metrics = MetricsCollector()
