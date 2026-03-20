import asyncio
import collections
import json
import logging
import os
import platform
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av
import cv2
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack, RTCRtpReceiver

# STUN: helps the server discover its public IP when deployed on EC2/cloud.
# On localhost both sides are 127.0.0.1 so STUN is a no-op but harmless.
# TURN is NOT needed — on EC2 UDP ports are open in the security group.
_ICE_SERVERS = RTCConfiguration(iceServers=[
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
])

MODEL_ENABLED = True

if MODEL_ENABLED:
    from core.proctor_coordinator import ProctorCoordinator
    from core.proctor_session import ProctorSession

from core.metrics import metrics as _metrics

# Logging is set up in main.py via setup_logging().
# Fall back to basicConfig so server.py also works when run directly.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

coordinator:     "ProctorCoordinator | None" = None
_session_config: dict                        = {}
MAX_CONNECTIONS: int                         = 40


def _build_config() -> dict:
    import config as _cfg_module
    return {k: getattr(_cfg_module, k) for k in dir(_cfg_module) if k.isupper()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global coordinator, _session_config, MAX_CONNECTIONS
    if MODEL_ENABLED:
        _session_config = _build_config()
        MAX_CONNECTIONS = _session_config.get("MAX_SESSIONS", 40)
        # CLI overrides take precedence over config.py values (set by main.py via env vars)
        _device       = os.getenv("PROCTOR_DEVICE")  or _session_config.get("YOLO_DEVICE",        "cpu")
        _half         = os.getenv("PROCTOR_HALF")    == "1" if os.getenv("PROCTOR_HALF") else _session_config.get("YOLO_HALF",         False)
        _warmup       = int(os.getenv("PROCTOR_WARMUP")) if os.getenv("PROCTOR_WARMUP") else _session_config.get("YOLO_WARMUP_FRAMES", 0)
        _min_vram     = _session_config.get("YOLO_MIN_VRAM_GB",    1.5)
        _imgsz        = _session_config.get("YOLO_IMGSZ",          640)
        _mp_stride    = _session_config.get("MEDIAPIPE_STRIDE",     1)

        coordinator = ProctorCoordinator(
            model_path        = _session_config["YOLO_MODEL_PATH"],
            max_sessions      = MAX_CONNECTIONS,
            tick_rate         = _session_config.get("TICK_RATE", 10),
            device            = _device,
            default_conf      = _session_config.get("YOLO_DEFAULT_CONF", 0.50),
            person_conf       = _session_config.get("YOLO_PERSON_CONF",  0.30),
            phone_conf        = _session_config.get("YOLO_PHONE_CONF",   0.65),
            book_conf         = _session_config.get("YOLO_BOOK_CONF",    0.70),
            audio_conf        = _session_config.get("YOLO_AUDIO_CONF",   0.41),
            half              = _half,
            warmup_frames     = _warmup,
            min_vram_gb       = _min_vram,
            imgsz             = _imgsz,
            mediapipe_stride  = _mp_stride,
        )
        await coordinator.start()
        logger.info("ProctorCoordinator started (max=%d  device=%s  half=%s  warmup=%d)",
                    MAX_CONNECTIONS, _device, _half, _warmup)
    yield
    if coordinator is not None:
        await coordinator.stop()
    _metrics.stop()


app = FastAPI(lifespan=lifespan)

# ── Request tracking middleware ───────────────────────────────────────────────
# Normalises path parameters so /snapshot/abc123 → /snapshot/{pc_id}
# and records count + latency per normalised endpoint.

_PATH_PARAMS = [
    # (prefix, replacement)
    ("/snapshot/",   "/snapshot/{pc_id}"),
    ("/stream/",     "/stream/{pc_id}"),
    ("/risk/",       "/risk/{pc_id}"),
    ("/alerts/",     "/alerts/{pc_id}"),
    ("/tab_switch/", "/tab_switch/{pc_id}"),
    ("/debug/",      "/debug/{pc_id}"),
    ("/session/",    "/session/{pc_id}/log"),
    ("/report/",     "/report/{report_id}"),
    ("/proof/",      "/proof/{path}"),
]


def _normalise_path(path: str) -> str:
    for prefix, template in _PATH_PARAMS:
        if path.startswith(prefix):
            return template
    return path


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        latency  = (time.perf_counter() - t0) * 1000
        endpoint = f"{request.method} {_normalise_path(request.url.path)}"
        _metrics.record_request(endpoint, response.status_code, latency)
        return response


app.add_middleware(RequestMetricsMiddleware)

# ── CORS — allow NextJS dev server and any configured origin ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler — ensures CORS headers on 500 errors ────────────
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )

pcs: set = set()
stream_stats: dict[str, dict] = {}
snapshots:    dict[str, bytes] = {}
fps_log:      list[dict]       = []
_device_counter = 0
_pcs_by_id: dict[str, RTCPeerConnection] = {}   # pc_id → pc (for trickle ICE)


# ── Tracks ────────────────────────────────────────────────────────────────────

class VideoAnalyzerTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, pc_id, stats, label, session=None):
        super().__init__()
        self.track   = track
        self.pc_id   = pc_id
        self.stats   = stats
        self.label   = label
        self.session = session

        self._frame_times: collections.deque = collections.deque(maxlen=60)
        self._last_log      = time.time()
        self._last_sample   = time.time()
        self._last_snapshot = 0.0
        self._frame_count   = 0

    async def recv(self):
        frame = await self.track.recv()
        now   = time.time()
        self._frame_times.append(now)
        self._frame_count += 1

        fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps  = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            intervals = [
                (self._frame_times[i] - self._frame_times[i - 1]) * 1000
                for i in range(1, len(self._frame_times))
            ]
            mean_iv = sum(intervals) / len(intervals)
            jitter  = (sum((x - mean_iv) ** 2 for x in intervals) / len(intervals)) ** 0.5
            self.stats.update({
                "fps"             : round(fps, 2),
                "jitter_ms"       : round(jitter, 2),
                "mean_interval_ms": round(mean_iv, 2),
                "total_frames"    : self.stats.get("total_frames", 0) + 1,
                "resolution"      : f"{frame.width}x{frame.height}",
            })
        else:
            self.stats["total_frames"] = self.stats.get("total_frames", 0) + 1

        try:
            img = frame.to_ndarray(format="bgr24")
            img = np.ascontiguousarray(img)
        except Exception as exc:
            logger.debug("[%s] frame decode error (skipping frame): %s", self.pc_id, exc)
            return frame  # keep latest_frame at last good value

        # Encode raw JPEG snapshot at ~5 Hz for admin live-view
        if now - self._last_snapshot >= 0.2:
            self._last_snapshot = now
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                snapshots[self.pc_id] = buf.tobytes()

        if fps > 0 and now - self._last_sample >= 2.0:
            self._last_sample = now
            fps_log.append({"t": round(now, 3), "fps": round(fps, 2),
                             "concurrent": len(pcs), "label": self.label})

        if self.session is not None:
            self.session.latest_frame = img
            self.session.observed_fps = fps if fps > 0 else self.session.observed_fps

        if now - self._last_log >= 5.0:
            self._last_log = now
            logger.info("[%s] fps=%.2f  jitter=%s ms  res=%s  concurrent=%d",
                        self.label, fps, self.stats.get("jitter_ms", "?"),
                        self.stats.get("resolution", "?"), len(pcs))
        return frame


class AudioAnalyzerTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track, stats, session=None):
        super().__init__()
        self.track      = track
        self.stats      = stats
        self.session    = session
        self._packet_times: collections.deque = collections.deque(maxlen=100)
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    async def recv(self):
        frame = await self.track.recv()
        now   = time.time()
        self._packet_times.append(now)

        if len(self._packet_times) >= 2:
            span = self._packet_times[-1] - self._packet_times[0]
            rate = (len(self._packet_times) - 1) / span if span > 0 else 0.0
            self.stats.update({
                "audio_packet_rate"  : round(rate, 2),
                "audio_total_packets": self.stats.get("audio_total_packets", 0) + 1,
                "audio_sample_rate"  : frame.sample_rate,
                "audio_channels"     : len(frame.layout.channels),
            })
        else:
            self.stats["audio_total_packets"] = self.stats.get("audio_total_packets", 0) + 1

        for r_frame in self._resampler.resample(frame):
            pcm       = r_frame.to_ndarray()
            pcm_int16 = pcm.astype(np.int16)
            if self.session is not None:
                self.session.push_audio_chunk(pcm_int16.T.flatten().tobytes(), now)
        return frame


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    return JSONResponse({
        pc_id: {k: v for k, v in s.items()}
        for pc_id, s in stream_stats.items()
    })

@app.get("/sessions")
async def list_sessions():
    """Active sessions list for the admin dashboard cards."""
    sessions_data = []
    for pc_id, stats in stream_stats.items():
        entry = {
            "pc_id"           : pc_id,
            "label"           : stats.get("label"),
            "connection_state": stats.get("connection_state"),
            "fps"             : stats.get("fps"),
            "resolution"      : stats.get("resolution"),
        }
        if MODEL_ENABLED and coordinator and pc_id in coordinator.sessions:
            session   = coordinator.sessions[pc_id]
            risk_info = session.risk.get_display()
            entry.update({
                "risk_score"      : risk_info["score"],
                "risk_state"      : risk_info["state"],
                "alert_count"     : len(session.alert_log),
                "warning_count"   : len(session.warning_log),
                "terminated"      : risk_info.get("terminated", False),
            })
        sessions_data.append(entry)
    return JSONResponse(sessions_data)

@app.get("/snapshot/{pc_id}")
async def get_snapshot(pc_id: str):
    # When debug mode is on, produce an annotated frame on the fly
    if MODEL_ENABLED and coordinator and pc_id in coordinator.sessions:
        session = coordinator.sessions[pc_id]
        if session.debug_mode:
            debug_frame = session.get_debug_frame()
            if debug_frame is not None:
                ok, buf = cv2.imencode(".jpg", debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    return Response(content=buf.tobytes(), media_type="image/jpeg")

    data = snapshots.get(pc_id)
    if not data:
        raise HTTPException(status_code=404, detail="No snapshot yet")
    return Response(content=data, media_type="image/jpeg")


@app.post("/debug/{pc_id}")
async def toggle_debug(pc_id: str, request: Request):
    """Toggle annotated debug overlay on the admin live snapshot."""
    if coordinator is None or pc_id not in coordinator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    body    = await request.json()
    enabled = bool(body.get("enabled", True))
    coordinator.sessions[pc_id].set_debug(enabled)
    return JSONResponse({"ok": True, "debug": enabled})

@app.get("/risk/{pc_id}")
async def get_risk(pc_id: str):
    if coordinator is None or pc_id not in coordinator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(coordinator.sessions[pc_id].risk.get_display())

@app.get("/session/{pc_id}/log")
async def get_session_log(pc_id: str):
    """Return full alert/warning history + current risk for a live session."""
    if coordinator is None or pc_id not in coordinator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = coordinator.sessions[pc_id]
    return JSONResponse({
        "alert_log"  : session.alert_log,
        "warning_log": session.warning_log,
        "risk"       : session.risk.get_display(),
    })

@app.post("/tab_switch/{pc_id}")
async def report_tab_switch(pc_id: str):
    """
    Called by the candidate frontend whenever they leave the exam tab.
    Scoring: occ=1 → warning, occ=2 → alert +15pts, occ>=3 → TERMINATED.
    """
    if coordinator is None or pc_id not in coordinator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = coordinator.sessions[pc_id]
    session.report_tab_switch(now=time.time())
    return JSONResponse({"ok": True, "risk": session.risk.get_display()})

@app.get("/alerts/{pc_id}")
async def get_alerts(pc_id: str):
    if coordinator is None or pc_id not in coordinator.sessions:
        return JSONResponse({"warnings": [], "alerts": []})
    session = coordinator.sessions[pc_id]
    return JSONResponse({
        "warnings": session._alert_manager.get_active_warnings(),
        "alerts"  : session._alert_manager.get_active_alerts(),
    })

@app.get("/capacity")
async def capacity():
    active = len(pcs)
    return JSONResponse({"active": active, "max": MAX_CONNECTIONS,
                         "available": active < MAX_CONNECTIONS})

@app.get("/coordinator/stats")
async def coordinator_stats():
    if coordinator is None:
        return JSONResponse({"enabled": False})
    return JSONResponse({"enabled": True, **coordinator.diagnostics()})


@app.get("/metrics")
async def get_metrics():
    """
    Live metrics snapshot — request counts, latency percentiles,
    session counts, alert/warning totals, YOLO timing, system resources.
    """
    snap = _metrics.snapshot()
    # Attach live coordinator diagnostics
    if coordinator is not None:
        snap["coordinator"].update(coordinator.diagnostics())
    return JSONResponse(snap)


@app.get("/system/report")
async def system_report():
    """
    Comprehensive system performance report.
    Combines metrics, coordinator diagnostics, active session details,
    hardware info, and runtime environment into one JSON document.
    """
    import torch

    snap    = _metrics.snapshot()
    now_utc = datetime.now(timezone.utc)
    proc    = psutil.Process()

    # ── Runtime environment ───────────────────────────────────────────────────
    env = {
        "python_version" : sys.version.split()[0],
        "platform"       : platform.platform(),
        "hostname"       : platform.node(),
        "pid"            : proc.pid,
        "cpu_count"      : psutil.cpu_count(logical=True),
        "cpu_phys"       : psutil.cpu_count(logical=False),
        "ram_total_gb"   : round(psutil.virtual_memory().total / 1e9, 1),
        "ram_avail_gb"   : round(psutil.virtual_memory().available / 1e9, 1),
    }

    # ── Detector / model info ─────────────────────────────────────────────────
    detector_info: dict = {"model_enabled": MODEL_ENABLED}
    if MODEL_ENABLED and coordinator is not None:
        detector_info.update(coordinator.detector.device_info)

    # ── Coordinator ───────────────────────────────────────────────────────────
    coord_info: dict = {"enabled": coordinator is not None}
    if coordinator is not None:
        coord_info.update(coordinator.diagnostics())
        coord_info.update(snap.get("coordinator", {}))

    # ── Active sessions summary ───────────────────────────────────────────────
    active_sessions = []
    if coordinator is not None:
        for pc_id, session in list(coordinator.sessions.items()):
            risk      = session.risk.get_display()
            s_stats   = stream_stats.get(pc_id, {})
            active_sessions.append({
                "pc_id"          : pc_id,
                "label"          : s_stats.get("label", "?"),
                "state"          : s_stats.get("connection_state", "?"),
                "fps"            : s_stats.get("fps"),
                "resolution"     : s_stats.get("resolution"),
                "alerts"         : len(session.alert_log),
                "warnings"       : len(session.warning_log),
                "risk_score"     : risk["score"],
                "risk_state"     : risk["state"],
                "terminated"     : risk["terminated"],
                "debug_mode"     : session.debug_mode,
            })

    # ── GPU info ──────────────────────────────────────────────────────────────
    gpu_info: dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info.update({
            "name"           : props.name,
            "vram_total_gb"  : round(props.total_memory / 1e9, 1),
            "vram_alloc_mb"  : round(torch.cuda.memory_allocated(0) / 1e6, 1),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 1),
            "compute_cap"    : f"{props.major}.{props.minor}",
            "torch_version"  : torch.__version__,
        })

    # ── Session config snapshot ───────────────────────────────────────────────
    safe_cfg = {
        k: v for k, v in _session_config.items()
        if isinstance(v, (int, float, bool, str))
    }

    report = {
        "generated_at"     : now_utc.isoformat(),
        "uptime"           : snap["uptime"],
        "uptime_s"         : snap["uptime_s"],
        "environment"      : env,
        "gpu"              : gpu_info,
        "detector"         : detector_info,
        "coordinator"      : coord_info,
        "requests"         : snap["requests"],
        "sessions"         : {
            **snap["sessions"],
            "max_concurrent" : MAX_CONNECTIONS,
            "active_details" : active_sessions,
        },
        "events"           : snap["events"],
        "yolo_performance" : snap["yolo"],
        "system_resources" : snap["system"],
        "config"           : safe_cfg,
    }

    return JSONResponse(report)


# ── SSE alert stream ───────────────────────────────────────────────────────────

@app.get("/stream/{pc_id}")
async def stream_alerts(pc_id: str):
    """
    Server-Sent Events stream for a candidate session.
    Pushes: alert | warning | risk_update | session_end events.
    Frontend connects once and receives all subsequent events.
    """
    async def generate():
        if coordinator is None or pc_id not in coordinator.sessions:
            yield f"data: {json.dumps({'type': 'error', 'message': 'session not found'})}\n\n"
            return

        session = coordinator.sessions[pc_id]
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        session.subscribe_sse(queue)

        # Send current state snapshot on connect
        yield f"data: {json.dumps({'type': 'connected', 'pc_id': pc_id, 'risk': session.risk.get_display()})}\n\n"

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue

                if event is None:        # sentinel — session ended
                    break

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") == "session_end":
                    break
        finally:
            session.unsubscribe_sse(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Proof file serving ────────────────────────────────────────────────────────

@app.get("/proof/{path:path}")
async def serve_proof(path: str):
    """Serve proof files (JPEG images, WAV audio) from the reports directory."""
    full_path = (Path("reports") / path).resolve()
    reports_root = Path("reports").resolve()

    # Safety: prevent path traversal outside reports/
    if not str(full_path).startswith(str(reports_root)):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Proof file not found")

    media_type = "audio/wav" if full_path.suffix == ".wav" else "image/jpeg"
    return FileResponse(str(full_path), media_type=media_type)


# ── Report endpoints ──────────────────────────────────────────────────────────

@app.get("/reports")
async def list_reports():
    """List all completed session report IDs."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return JSONResponse([])
    reports = []
    for d in sorted(reports_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "report.json").exists():
            reports.append(d.name)
    return JSONResponse(reports)

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Get the JSON report for a completed session."""
    # Sanitize: only allow directory names, no path separators
    if "/" in report_id or "\\" in report_id or ".." in report_id:
        raise HTTPException(status_code=400, detail="Invalid report ID")
    path = Path("reports") / report_id / "report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path) as f:
        return JSONResponse(json.load(f))


# ── Exam config (admin-controlled detection toggles) ─────────────────────────

@app.get("/exam/config")
async def get_exam_config():
    """
    Return current detection toggles.
    Shows global defaults merged with any admin overrides.
    """
    defaults = {k: v for k, v in _session_config.items() if k.startswith("DETECT_")}
    if coordinator is not None:
        defaults.update(coordinator.exam_config)
    return JSONResponse(defaults)

@app.post("/exam/config")
async def set_exam_config(request: Request):
    """
    Update detection toggles at runtime.
    Changes apply immediately to all active sessions.
    Disabled detections skip both computation and scoring — saves CPU/GPU.
    """
    body    = await request.json()
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not running")
    updated = coordinator.update_exam_config(body)
    return JSONResponse({"updated": updated})

@app.get("/analysis")
async def analysis():
    buckets: dict[int, list[float]] = {}
    for entry in fps_log:
        buckets.setdefault(entry["concurrent"], []).append(entry["fps"])
    by_concurrent = {
        c: {"avg_fps": round(sum(v) / len(v), 2), "min_fps": round(min(v), 2),
            "max_fps": round(max(v), 2), "samples": len(v)}
        for c, v in sorted(buckets.items())
    }
    return JSONResponse({"raw_log": fps_log, "by_concurrent": by_concurrent})


# ── WebRTC offer ──────────────────────────────────────────────────────────────

def _strip_rtx_from_sdp(sdp_str: str) -> str:
    """
    Remove all video/rtx payload-type references from an SDP string.

    Why: aiortc's codecs module has no handler for 'video/rtx'.  When the
    browser negotiates RTX (RTP retransmission, RFC 4588) alongside VP8,
    aiortc still accepts it and eventually starts a decoder_worker thread for
    the RTX payload type.  That thread crashes immediately:
        ValueError: No decoder found for MIME type `video/rtx`
    The crash kills the video-decoder thread, recv() blocks forever, and the
    frame feed freezes.

    Stripping RTX from the offer before setRemoteDescription means aiortc
    never creates RTX receivers, the answer SDP omits RTX, and the browser
    stops sending RTX packets — so the decoder thread is never triggered.
    """
    sep   = "\r\n" if "\r\n" in sdp_str else "\n"
    lines = sdp_str.split(sep)

    # Pass 1 — collect RTX payload types  (e.g. "a=rtpmap:97 rtx/90000")
    rtx_pts: set[str] = set()
    for line in lines:
        if line.startswith("a=rtpmap:"):
            rest = line[len("a=rtpmap:"):]
            pt, _, codec = rest.partition(" ")
            if codec.lower().startswith("rtx/"):
                rtx_pts.add(pt.strip())

    if not rtx_pts:
        return sdp_str   # nothing to strip

    logger.debug("Stripping RTX payload types from offer SDP: %s", rtx_pts)

    # Pass 2 — drop lines that reference RTX PTs, rewrite m= lines
    out: list[str] = []
    for line in lines:
        # Drop rtpmap / fmtp / rtcp-fb lines for RTX PTs
        if any(
            line.startswith(f"a=rtpmap:{pt} ")
            or line.startswith(f"a=fmtp:{pt} ")
            or line.startswith(f"a=rtcp-fb:{pt} ")
            for pt in rtx_pts
        ):
            continue
        # Rewrite "m=video ... PT1 PT2 PT3" to remove RTX PTs
        if line.startswith("m="):
            parts = line.split()
            kept  = [p for p in parts[3:] if p not in rtx_pts]
            out.append(" ".join(parts[:3] + kept))
        else:
            out.append(line)

    return sep.join(out)


@app.post("/ice-candidate/{pc_id}")
async def add_ice_candidate(pc_id: str, request: Request):
    """Receive a trickle ICE candidate from the browser and add it to the peer connection."""
    pc = _pcs_by_id.get(pc_id)
    if pc is None:
        raise HTTPException(status_code=404, detail="Session not found")

    params = await request.json()
    candidate_str = params.get("candidate", "")
    if not candidate_str:
        return {"ok": True}   # end-of-candidates signal — nothing to do

    try:
        from aiortc.sdp import candidate_from_sdp
        # Browser sends "candidate:..." — strip the prefix for the parser
        sdp_value = candidate_str[len("candidate:"):] if candidate_str.startswith("candidate:") else candidate_str
        ice_candidate = candidate_from_sdp(sdp_value)
        ice_candidate.sdpMid       = params.get("sdpMid")
        ice_candidate.sdpMLineIndex = params.get("sdpMLineIndex", 0)
        await pc.addIceCandidate(ice_candidate)
    except Exception as exc:
        logger.warning("Failed to add ICE candidate for %s: %s", pc_id, exc)

    return {"ok": True}


@app.post("/offer")
async def offer(request: Request):
    if len(pcs) >= MAX_CONNECTIONS:
        return JSONResponse(
            {"error": f"Server at capacity ({MAX_CONNECTIONS} sessions)."},
            status_code=503,
        )

    global _device_counter
    _device_counter += 1
    device_label = f"Candidate_{_device_counter}"

    params = await request.json()
    # Strip video/rtx before aiortc sees the offer — prevents the decoder_worker
    # thread from crashing on "No decoder found for MIME type `video/rtx`" which
    # kills all video decoding and freezes the frame feed.
    clean_sdp = _strip_rtx_from_sdp(params["sdp"])
    offer_sdp = RTCSessionDescription(sdp=clean_sdp, type=params["type"])

    # Per-session detection overrides (from exam-level config set by admin)
    detection_override = params.get("detection_config", {})

    pc    = RTCPeerConnection(configuration=_ICE_SERVERS)
    pc_id = str(id(pc))
    pcs.add(pc)
    _pcs_by_id[pc_id] = pc
    stream_stats[pc_id] = {"label": device_label, "connection_state": "new"}
    _metrics.inc_session()

    ts          = time.strftime("%Y%m%d_%H%M%S")
    file_prefix = f"{ts}_{device_label}"

    session: "ProctorSession | None" = None
    if MODEL_ENABLED and coordinator is not None:
        session_cfg = {**_session_config, **detection_override}
        reports_dir = Path("reports") / file_prefix
        session = ProctorSession(
            session_id       = device_label,
            session_dir      = reports_dir,
            config           = session_cfg,
        )
        coordinator.add_session(pc_id, session)
        logger.info("[%s] ProctorSession created (report_id=%s)", device_label, file_prefix)

    video_track: VideoAnalyzerTrack | None = None
    audio_track: AudioAnalyzerTrack | None = None

    logger.info("[%s] New connection — %d/%d slots", device_label, len(pcs), MAX_CONNECTIONS)

    @pc.on("track")
    async def on_track(track):
        nonlocal video_track, audio_track
        if track.kind == "video":
            video_track = VideoAnalyzerTrack(
                track, pc_id, stream_stats[pc_id], device_label, session=session,
            )
            async def _video_loop():
                """
                Uses asyncio.wait() — NOT asyncio.wait_for() — so a stalled aiortc
                jitter-buffer never silently blocks this loop.
                asyncio.wait_for cancels the recv() task and corrupts aiortc internal
                jitter state; asyncio.wait simply returns after the timeout leaving the
                task alive so we can keep waiting on it without any corruption.
                """
                recv_task  = asyncio.ensure_future(video_track.recv())
                stall_n    = 0

                while True:
                    done, _ = await asyncio.wait({recv_task}, timeout=10.0)

                    if recv_task in done:
                        stall_n = 0
                        exc = recv_task.exception()
                        if exc is not None:
                            exc_name = type(exc).__name__
                            if exc_name == "MediaStreamError" or "ended" in str(exc).lower():
                                logger.info("[%s] video track ended: %s", device_label, exc)
                                break
                            logger.warning("[%s] video recv error (%s): %s",
                                           device_label, exc_name, exc)
                            await asyncio.sleep(0.033)
                        recv_task = asyncio.ensure_future(video_track.recv())

                    else:
                        stall_n += 1
                        logger.warning(
                            "[%s] video recv stalled >%ds (stall #%d) — "
                            "pc_state=%s  track_ready=%s",
                            device_label, 10 * stall_n, stall_n,
                            pc.connectionState,
                            getattr(video_track.track, "readyState", "?"),
                        )
                        # Send RTCP PLI to browser — requests a fresh keyframe which
                        # can unblock the jitter buffer.  The task stays alive (no cancel).
                        try:
                            for _t in pc.getTransceivers():
                                if _t.kind == "video":
                                    _r    = _t.receiver
                                    _pfn  = getattr(_r, "_send_pli", None) \
                                            or getattr(_r, "send_pli", None)
                                    if _pfn:
                                        asyncio.ensure_future(_pfn())
                        except Exception:
                            pass
                        # Keep waiting on the same recv_task — do NOT cancel it

                logger.info("[%s] _video_loop exited", device_label)
            asyncio.ensure_future(_video_loop())

        elif track.kind == "audio":
            audio_track = AudioAnalyzerTrack(track, stream_stats[pc_id], session=session)
            async def _audio_loop():
                recv_task = asyncio.ensure_future(audio_track.recv())
                while True:
                    done, _ = await asyncio.wait({recv_task}, timeout=15.0)
                    if recv_task in done:
                        exc = recv_task.exception()
                        if exc is not None:
                            exc_name = type(exc).__name__
                            if exc_name == "MediaStreamError" or "ended" in str(exc).lower():
                                logger.info("[%s] audio track ended: %s", device_label, exc)
                                break
                            logger.warning("[%s] audio recv error (%s): %s",
                                           device_label, exc_name, exc)
                            await asyncio.sleep(0.033)
                        recv_task = asyncio.ensure_future(audio_track.recv())
                    else:
                        logger.debug("[%s] audio recv stalled >15s — pc_state=%s",
                                     device_label, pc.connectionState)
                        # Keep waiting on same task
                logger.info("[%s] _audio_loop exited", device_label)
            asyncio.ensure_future(_audio_loop())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        if pc_id in stream_stats:
            stream_stats[pc_id]["connection_state"] = state
        logger.info("[%s] state → %s", device_label, state)

        if state in ("failed", "closed", "disconnected"):
            if MODEL_ENABLED and coordinator is not None:
                coordinator.remove_session(pc_id)
            stream_stats.pop(pc_id, None)
            snapshots.pop(pc_id, None)
            pcs.discard(pc)
            _pcs_by_id.pop(pc_id, None)
            _metrics.dec_session()
            logger.info("[%s] ended — %d/%d slots", device_label, len(pcs), MAX_CONNECTIONS)

    await pc.setRemoteDescription(offer_sdp)

    # Force VP8 — aiortc's native codec, far more resilient to packet loss than
    # H.264 on internet paths.  H.264 P-frames create long dependency chains: one
    # lost UDP packet corrupts all subsequent frames until the next I-frame and
    # can stall the aiortc jitter buffer indefinitely.  VP8 has no such chain.
    try:
        capabilities = RTCRtpReceiver.getCapabilities("video")
        if capabilities:
            vp8_codecs = [c for c in capabilities.codecs
                          if c.mimeType.lower() == "video/vp8"]
            if vp8_codecs:
                for transceiver in pc.getTransceivers():
                    if transceiver.kind == "video":
                        transceiver.setCodecPreferences(vp8_codecs)
                        logger.info("[%s] codec forced to VP8", device_label)
    except Exception as exc:
        logger.warning("[%s] VP8 codec preference failed (falling back): %s",
                       device_label, exc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp"         : pc.localDescription.sdp,
        "type"        : pc.localDescription.type,
        "device_id"   : pc_id,
        "device_label": device_label,
        "report_id"   : file_prefix,
    })
