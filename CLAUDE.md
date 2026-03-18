# AI Proctor — Complete Project Documentation

> **Read this first in any new session.** This file covers the full system architecture,
> every implementation decision made, all known bugs and fixes, resource profiling results,
> and exactly what needs to be done next for the GPU-hosted production version.

---

## 1. Project Structure

```
NextAndFastApiWithWebRTC/
├── Proctor-webRTC/          ← FastAPI + WebRTC backend (Python)
│   ├── main.py              ← Entry point with argparse CLI
│   ├── server.py            ← FastAPI app, WebRTC offer handling, all REST endpoints
│   ├── config.py            ← All tunable constants (thresholds, toggles, device config)
│   ├── core/
│   │   ├── proctor_coordinator.py   ← Orchestrates all sessions, owns YOLO, tick loop
│   │   ├── proctor_session.py       ← Per-candidate state machine
│   │   ├── risk_engine.py           ← Score accumulation, state machine, termination
│   │   ├── alert_engine.py          ← Alert/warning generation with cooldowns
│   │   ├── head_tracker.py          ← Duration-gating for head conditions
│   │   ├── liveness.py              ← Fake-presence / no-movement detector
│   │   ├── audio_monitor.py         ← WebRTC PCM audio ring buffer + VAD
│   │   ├── object_tracker.py        ← Temporal vote-window for YOLO objects
│   │   └── metrics.py               ← In-process metrics singleton (CPU/RAM/GPU/latency)
│   ├── detectors/
│   │   ├── object_detector.py       ← YOLO wrapper (batch inference, device selection)
│   │   ├── head_pose_detector.py    ← MediaPipe head pose, gaze, EAR/blink
│   │   ├── lip_detector.py          ← MediaPipe lip MAR, speaking/yawning detection
│   │   └── face_mesh_provider.py    ← Shared FaceMesh instance per session
│   ├── settings/
│   │   ├── scoring.py       ← All risk scoring values (scores, cooldowns, thresholds)
│   │   └── alerts.py        ← Alert/warning message strings and cooldowns
│   └── utils/
│       ├── draw.py          ← CV2 drawing: bboxes, alerts, MIC indicator
│       ├── logging_config.py ← Structured JSON logging with rotation
│       ├── alerts.py        ← AlertManager class
│       └── proof_writer.py  ← JPEG proof capture + WAV audio clip on alert
│
└── frontend/                ← Next.js 14 admin + candidate UI
    ├── app/
    │   ├── admin/
    │   │   ├── page.tsx             ← Admin dashboard (session cards)
    │   │   ├── session/[pc_id]/     ← Live session view with debug toggle
    │   │   └── monitor/page.tsx     ← System Monitor (Live Metrics + System Report)
    │   └── candidate/page.tsx       ← Candidate exam page
    ├── lib/
    │   ├── api.ts           ← All backend API calls
    │   └── types.ts         ← TypeScript interfaces
    └── components/
```

---

## 2. Full System Architecture

### 2.1 WebRTC + Inference Pipeline

```
Candidate Browser
  └── getUserMedia() → WebRTC offer → POST /offer
        ↓
server.py: RTCPeerConnection
  └── VideoAnalyzerTrack.recv()
        - Decode AV frame → BGR numpy (h×w×3)
        - session.latest_frame = img        (reference swap, no copy)
        - session.observed_fps = fps
        - JPEG encode at 2 Hz → snapshots[pc_id]  (admin live-view)
        ↓
ProctorCoordinator._tick_loop()  [asyncio task, 10 Hz target]
  └── _tick()
        Step 1: snapshot {pc_id: session} for all non-terminated sessions with frames
        Step 2: YOLO batch — await loop.run_in_executor(None, detect_batch, frames)
                  → one forward pass for ALL N sessions simultaneously
                  → returns list[list[dict]]  (one detection list per frame)
        Step 3: MediaPipe — per-session, only if session.needs_mediapipe()
                  → loop.run_in_executor(_mp_pool, session.run_mediapipe, frame)
                  → if not needed: immediately-resolved asyncio.Future(_NULL_MP)
                  → FaceMesh → HeadPoseDetector → LipDetector
        Step 4: session.update(detections, mp_result, frame, now, fps)
                  → risk scoring, alert generation, SSE push
```

### 2.2 Detection → Alert → Risk Flow

```
YOLO detections (person, cell_phone, book, headphone, earbud)
    ↓
ObjectTemporalTracker  (vote window: N detections in last W frames)
    ↓
RiskEngine.process_event(key, triggered, confidence)
    ↓
AlertEngine._handle_event(rev, frame, now)
    ↓
SSE push to all subscribers   ← admin frontend listens on GET /stream/{pc_id}
```

### 2.3 MediaPipe Flow

```
FaceMeshProvider.process(frame)   ← one FaceMesh call, landmarks reused by both detectors
    ├── HeadPoseDetector.detect(frame, landmarks=landmarks)
    │     → (looking_away, looking_down, looking_up, looking_left, looking_right,
    │        partial_face, yaw, pitch, gaze, ear, blinked, total_blinks)
    └── LipDetector.process(frame, ts, landmarks=landmarks)
          → LipState(mar, is_open, is_speaking, is_yawning, face_detected)
```

---

## 3. Key Implementation Details

### 3.1 YOLO Inference — Critical Rule

**Never pass `imgsz` as a runtime argument to the model call.**

```python
# CORRECT — output bbox coords are always in original frame pixel space
results = self.model(frames, verbose=False, device=self.device)

# WRONG — breaks orig_shape tracking in batch mode, bboxes land in wrong positions
results = self.model(frames, verbose=False, device=self.device, imgsz=320)
```

Ultralytics YOLO, when called with a list of numpy arrays (batch mode), stores `orig_shape`
per frame and applies the inverse letterbox transform automatically. Passing `imgsz` at runtime
breaks this in batch mode in some Ultralytics versions, causing all bboxes to be in the
letterboxed coordinate space rather than the original frame space. The result is that bounding
boxes appear in the wrong position in the debug overlay.

**Correct approach for GPU speed**: set `imgsz` at model *load time* if you want 640→320
resize, not at inference time. Or pre-resize frames manually and scale coords back explicitly.
But the simplest correct approach is no `imgsz` override — let YOLO use its native 640.

### 3.2 MediaPipe Conditional Execution

MediaPipe (FaceMesh + HeadPoseDetector + LipDetector) is **only** run when at least one of
these detection flags is enabled:

```python
_MEDIAPIPE_FLAGS = {
    "DETECT_LOOKING_AWAY", "DETECT_LOOKING_DOWN", "DETECT_LOOKING_UP",
    "DETECT_LOOKING_SIDE", "DETECT_PARTIAL_FACE", "DETECT_FAKE_PRESENCE",
    "DETECT_FACE_HIDDEN", "DETECT_SPEAKER_AUDIO",
}
```

`session.needs_mediapipe()` is checked **before** dispatching to the thread pool. If False,
an immediately-resolved `asyncio.Future` with null results is used — zero thread-pool overhead,
zero C++ compute. This is important because the detection toggles are runtime-configurable by
the admin (via `POST /exam/config`), so a full exam with only object detection enabled will
save the entire MediaPipe compute budget per tick.

### 3.3 Tab Switch / Focus Violation Scoring

- **Every** tab switch (including the first) immediately adds `TAB_SWITCH_SCORE = 15` fixed
  (non-decaying) points. There is no grace on the first occurrence.
- At `TAB_SWITCH_TERMINATE_COUNT = 3` total switches → exam is auto-terminated.
- Candidate frontend also detects `window.blur` (split-screen focus loss) using a 150ms
  `setTimeout` to distinguish from tab switch (which fires `visibilitychange` first).
- Copy/paste/cut are globally disabled on the candidate page during an active exam session.
- `fetch(..., keepalive: true)` is used for `POST /tab_switch/{pc_id}` so it completes even
  when the browser tab becomes hidden.

### 3.4 Debug Overlay (Admin Side)

`GET /snapshot/{pc_id}` returns:
- **Normal mode**: raw JPEG from WebRTC stream (encoded at 2 Hz)
- **Debug mode** (`session.debug_mode = True`): `session.get_debug_frame()` is called,
  which composites on the latest frame:
  1. YOLO bounding boxes (`draw_detections`)
  2. MediaPipe landmarks, head-pose lines, gaze dots, EAR value (`head_detector.draw_debug`)
  3. Lip MAR contour + speaking indicator (`lip_detector._draw_overlay`)
  4. Active alert/warning banners (`draw_alerts`)
  5. Audio MIC indicator (`draw_audio_status`)
  6. Risk score panel top-right (`_draw_risk_panel`)

Toggle via `POST /debug/{pc_id}` with body `{"enabled": true/false}`.

### 3.5 Risk Scoring Architecture

Two score buckets:
- **Fixed (non-decaying)**: tab_switch, phone, fake_presence, multiple_people, no_person
- **Decaying**: gaze events, book, headphone, earbud, partial_face, face_hidden, speaker_audio

State machine: `NORMAL → WARNING (30pts) → HIGH_RISK (60pts) → ADMIN_REVIEW (100pts) → TERMINATED`

Termination triggers:
- Tab switches >= 3
- Multiple people continuously >= 20 s
- No person continuously >= 20 s
- Any explicit admin termination

All scoring values in `settings/scoring.py`. Alert messages and cooldowns in `settings/alerts.py`.

### 3.6 Production Logging

`utils/logging_config.py` sets up:
- `logs/app.log` — JSON structured, rotating 10 MB × 5 files
- `logs/error.log` — JSON structured, rotating 5 MB × 3 files, ERROR+ only
- Console — colour ANSI readable format
- Noisy third-party loggers suppressed: aiortc, aioice, ultralytics, asyncio

Must be called as the **very first thing** in `main.py` before any other import.

### 3.7 Metrics & Monitoring

`core/metrics.py` — module-level singleton `metrics`.

Tracks: request count/latency (p50/p95/p99) per normalised endpoint, session counts,
alert/warning totals, YOLO batch latency, tick latency, CPU%, process RSS MB,
GPU util% (pynvml), GPU VRAM (torch).

Two endpoints:
- `GET /metrics` — live snapshot JSON
- `GET /system/report` — full report: environment, GPU, detector, coordinator, active sessions,
  all config values

Frontend: `frontend/app/admin/monitor/page.tsx` — two-tab page (Live Metrics auto-refresh 3s,
System Report on-demand). Accessible from admin dashboard nav.

---

## 4. All API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/offer` | WebRTC handshake — creates RTCPeerConnection and ProctorSession |
| GET | `/snapshot/{pc_id}` | JPEG frame (raw or debug-annotated) |
| POST | `/debug/{pc_id}` | Toggle debug overlay `{"enabled": bool}` |
| GET | `/stream/{pc_id}` | SSE stream: alert, warning, risk_update, session_end events |
| GET | `/risk/{pc_id}` | Current risk state JSON |
| GET | `/alerts/{pc_id}` | Active warnings and alerts |
| POST | `/tab_switch/{pc_id}` | Report tab switch / focus violation from candidate |
| GET | `/session/{pc_id}/log` | Full alert+warning history + current risk |
| GET | `/sessions` | List of all active sessions with risk summary |
| GET | `/metrics` | Live metrics snapshot |
| GET | `/system/report` | Full system performance report |
| GET | `/exam/config` | Current detection toggle state |
| POST | `/exam/config` | Update detection toggles at runtime |
| GET | `/reports` | List completed session report IDs |
| GET | `/report/{report_id}` | Get JSON for a completed session |
| GET | `/proof/{path}` | Serve proof JPEG/WAV files |
| GET | `/stats` | WebRTC stream stats |
| GET | `/capacity` | Current slot usage |
| GET | `/coordinator/stats` | Coordinator diagnostics |

---

## 5. CLI Usage

```bash
# Default (CPU, auto device selection)
python main.py

# Force CPU explicitly
python main.py --device cpu

# Force GPU (will warn if VRAM low)
python main.py --device cuda

# GPU with FP16 half-precision (~2x throughput, ~½ VRAM)
python main.py --device cuda --half

# GPU with CUDA kernel warmup (3 dummy passes at startup, in background thread)
python main.py --device cuda --half --warmup 3

# Custom port
python main.py --port 9000
```

CLI args are passed to `server.py` via environment variables:
- `PROCTOR_DEVICE` → `"auto" | "cuda" | "cpu"`
- `PROCTOR_HALF` → `"1" | "0"`
- `PROCTOR_WARMUP` → integer string

---

## 6. Device / GPU Configuration

### 6.1 `_resolve_device()` Logic

```
device="cpu"   → force CPU, no check
device="cuda"  → force GPU, warn if free VRAM < min_vram_gb
device="auto"  → CUDA if available AND free VRAM >= min_vram_gb, else CPU
```

`YOLO_MIN_VRAM_GB = 1.5` — minimum free VRAM for auto-selection.

### 6.2 FP16 Half-Precision

- Only enabled on CUDA (`self._half = half and self.device == "cuda"`)
- CPU FP16 is slower than FP32 — flag is silently ignored on CPU
- Activated via `model.half()` after loading

### 6.3 Warmup

- Default: `YOLO_WARMUP_FRAMES = 0` (disabled)
- When enabled, runs N dummy `model([zeros_frame])` passes in a **daemon background thread**
  via `_schedule_warmup()` — never blocks the asyncio event loop or server startup
- Purpose: pre-compile CUDA kernels so the first real frames don't have latency spikes

---

## 7. Resource Profiling — Single User on Local CPU Machine

Data from `resourceUtilizationReport.pdf` (3-minute session, 1 user, CPU mode):

| Metric | Value |
|--------|-------|
| CPU usage | 29.7% |
| Process RSS | 1245 MB |
| System available RAM | 1867 MB (622 MB headroom) |
| YOLO avg latency | 94.2 ms |
| YOLO P95 | 132.3 ms |
| YOLO P99 | 153.3 ms |
| Tick avg | 101.7 ms (target 100 ms) |
| Tick P95 | 139.7 ms |
| /snapshot poll rate | 3.2 req/s (admin frontend) |

### Concurrency Limits (CPU)

| Users | Est. tick | Est. Hz | CPU est. | RAM est. | Safe? |
|-------|-----------|---------|----------|----------|-------|
| 1 | 102 ms | 9.8 | 30% | 1245 MB | ✓ |
| 2 | ~185 ms | 5.4 | ~52% | ~1420 MB | ✓ |
| 3 | ~280 ms | 3.6 | ~76% | ~1580 MB | ⚠ viable |
| 4 | ~370 ms | 2.7 | ~98% | ~1740 MB | ✗ freeze |

**`MAX_SESSIONS = 3`** is set in `config.py` for the CPU build.

At 3.6 Hz the proctoring still works correctly — duration gates are 1.5–2 s, giving 5–7
samples per alert window even at 3.6 Hz.

---

## 8. Known Bugs Fixed

### 8.1 Bounding Box Misalignment (CRITICAL — was broken, now fixed)

**Root cause**: Earlier code manually resized frames to 320px before YOLO AND also passed
`imgsz=320` to the model call. This caused double coordinate transform (manual resize scale +
YOLO's internal inverse letterbox on top). Result: all bboxes were in the wrong position.

**Fix**: Never pre-resize frames. Never pass `imgsz` at runtime. Pass original frames to YOLO
and let it handle letterboxing internally. Output coords are always in original pixel space.

### 8.2 Backend Freeze on Startup

**Root cause**: `half=True` called `model.half()` on GPU, and `warmup_frames=3` ran 3 sync
YOLO passes inside `ObjectDetector.__init__()` — all during FastAPI lifespan startup,
blocking the asyncio event loop.

**Fix**: `half=False` and `warmup_frames=0` as defaults. Warmup moved to `_schedule_warmup()`
which starts a daemon thread.

### 8.3 Tab Switch Count Not Terminating (was 7/3 without terminating)

**Root cause**: `POST /tab_switch/{pc_id}` was not completing when the browser tab went
hidden — browser was suspending the fetch.

**Fix**: `fetch(..., keepalive: true)` in the candidate frontend.

### 8.4 Alert Flooding on Termination

**Root cause**: After termination, the tick loop kept processing the session and firing alerts.

**Fix**: `if self.risk.terminated: return` as first line of `session.update()`. Also added
`_termination_sse_sent` flag to send the `session_end` SSE event exactly once.

### 8.5 Face Hidden / No Person Collision

**Root cause**: Three conditions were triggering on a blank/covered camera — no_person,
face_hidden, and fake_presence all firing simultaneously.

**Fix**:
- `no_person`: YOLO sees 0 people AND no landmarks active
- `face_hidden`: YOLO sees person BUT no landmarks AND face was seen recently (recency gate)
- `fake_presence`: requires `landmarks_active = True` (real face data, not nulls)

---

## 9. What Needs to Be Done — GPU Production Build

This is the task list for the new GPU-hosted folder.

### 9.1 config.py — Change These Values

```python
# Increase from 3 (CPU limit) to what GPU can handle
MAX_SESSIONS = 20          # start here, tune based on profiling

# Keep 10 Hz — GPU can handle it
TICK_RATE = 10

# Change device to cuda
YOLO_DEVICE = "cuda"

# Enable FP16 — ~2x throughput on GPU, negligible accuracy loss
YOLO_HALF = True

# Enable warmup — pre-compile CUDA kernels at startup
YOLO_WARMUP_FRAMES = 3

# Increase VRAM gate if your GPU has more
YOLO_MIN_VRAM_GB = 2.0
```

### 9.2 object_detector.py — GPU Optimisations

1. **Persistent model on GPU** — model is already moved with `model.to(self.device)` and
   `model.half()`. No changes needed here.

2. **Enable `imgsz=320` correctly for GPU** — do NOT pass at inference time (see §3.1).
   Instead, if you want 320px inference for throughput, set it at load time:
   ```python
   self.model = YOLO(model_path)
   self.model.overrides['imgsz'] = 320   # sets default for all calls
   ```
   This way inference uses 320px but coordinate transforms work correctly.

3. **Batch size tuning** — GPU batching has superlinear efficiency gains unlike CPU.
   A batch of 10 frames on GPU may take only 30–40ms vs 10×30ms sequentially.
   Profile with `detect_batch()` at batch sizes 1, 5, 10, 20 and find the sweet spot
   for your GPU's memory bandwidth.

4. **Stream-based inference (optional)** — for very high concurrency, consider
   `torch.cuda.Stream` to overlap data transfer and compute.

### 9.3 proctor_coordinator.py — Thread Pool Sizing

```python
# On GPU, YOLO is fast (20-40ms). MediaPipe still runs on CPU in parallel.
# Increase thread pool for more concurrent MediaPipe calls.
_mp_workers = min(max(max_sessions, 8), 16)  # was capped at 4 for CPU
```

Also: on GPU the YOLO batch completes in ~20ms, then all N MediaPipe calls start in parallel.
If MediaPipe takes 15ms and there are 10 sessions, all 10 run in parallel in the thread pool
(Python GIL is released during MediaPipe's C++ compute). So the bottleneck becomes
`max(YOLO_batch_ms, max_single_MediaPipe_ms)` which is excellent.

### 9.4 server.py — Snapshot Rate

With GPU, YOLO finishes faster, so admin snapshots can be served at higher rate.
Currently set to 0.5s (2 Hz) in `VideoAnalyzerTrack.recv()`. For GPU:

```python
if now - self._last_snapshot >= 0.2:   # 5 Hz for GPU (was 0.5s/2Hz for CPU)
```

### 9.5 main.py — Production Server Settings

For a GPU server deployment, consider:
```bash
python main.py --device cuda --half --warmup 3 --port 8000
```

Also consider:
- Running behind nginx as a reverse proxy (handle TLS termination, WebSocket upgrades)
- `gunicorn -w 1 -k uvicorn.workers.UvicornWorker server:app` — single worker is mandatory
  because model and WebRTC peer connections are process-local
- Systemd service file for auto-restart

### 9.6 MediaPipe on GPU (Optional Enhancement)

MediaPipe FaceMesh currently runs on CPU even in GPU mode. For very high concurrency (20+
sessions), MediaPipe becomes the bottleneck. Options:
- Use `mediapipe` GPU delegate if available on Linux
- OR replace MediaPipe with a CUDA-based face landmark model (e.g., InsightFace on GPU)
- OR reduce FaceMesh resolution: `min_detection_confidence=0.7` reduces CPU time

### 9.7 YOLO Model Optimisation

Consider exporting the model to TensorRT for maximum GPU throughput:
```python
from ultralytics import YOLO
model = YOLO("finalBestV5.pt")
model.export(format="engine", device=0, half=True, imgsz=640)
# Then load as:
model = YOLO("finalBestV5.engine")
```
TensorRT typically gives 3–5× speedup vs PyTorch on the same GPU.

Note: `_supports_batch` in `ObjectDetector` already handles ONNX. Add TensorRT detection:
```python
self._supports_batch = not str(model_path).lower().endswith((".onnx", ".engine"))
# OR test if engine supports batch at init
```
Actually, TensorRT engines exported by Ultralytics DO support batch — just test at your
target batch size and update the flag logic.

### 9.8 Concurrency Targets for GPU

Based on CPU profiling (1 user = 94ms YOLO on CPU):
- GPU YOLO for 1 frame: estimate 5–15ms (RTX 3060 class) to 3–8ms (A100 class)
- GPU YOLO batch of 10: estimate 15–30ms total (batching is efficient on GPU)
- Expected GPU tick latency: 20–40ms for 10 users
- Expected max sessions: 15–25 concurrent (limited by MediaPipe CPU cost, not YOLO)

Profile with the `/metrics` endpoint to find actual GPU numbers.

---

## 10. Frontend — Admin Pages

### Admin Dashboard (`/admin`)
- Session cards with live risk scores and connection state
- Link to Monitor page (pulsing green dot)

### Session Detail (`/admin/session/[pc_id]`)
- Live snapshot polling (auto-refreshes)
- Debug ON/OFF toggle button — calls `POST /debug/{pc_id}`
- Risk display, alert/warning log
- SSE connection for live alerts

### Monitor (`/admin/monitor`)
- **Live Metrics tab** (auto-refresh 3s): CPU/RAM/GPU gauges, YOLO/tick latency cards,
  request breakdown table
- **System Report tab** (on-demand): full environment, hardware, active sessions, config grid,
  download JSON button

### Candidate Page (`/candidate`)
- Tab switch detection via `visibilitychange`
- Window focus loss via `window.blur` (150ms delay to de-duplicate from tab switch)
- Copy/paste/cut disabled during active session
- Demo exam question card + answer textarea for demonstration
- Risk panel shows "Focus violations" (covers both tab switch + window blur)

---

## 11. File-by-File Change Summary

| File | What changed |
|------|-------------|
| `config.py` | Added `YOLO_HALF`, `YOLO_WARMUP_FRAMES`, `YOLO_MIN_VRAM_GB`. Set `YOLO_DEVICE="cpu"`, `MAX_SESSIONS=3`, `TICK_RATE=10` |
| `main.py` | Added argparse: `--device`, `--half`, `--warmup`, `--port`. Sets env vars for server.py. Calls `setup_logging()` first. |
| `server.py` | Added `RequestMetricsMiddleware`, `POST /debug`, `GET /metrics`, `GET /system/report`. Reads env vars in lifespan. Snapshot rate 2Hz. |
| `detectors/object_detector.py` | Added `_resolve_device()` with VRAM check, `half`/`warmup_frames`/`min_vram_gb` params, `_schedule_warmup()` daemon thread. **Never passes `imgsz` to model call.** |
| `core/proctor_coordinator.py` | Added `min_vram_gb` param, capped MediaPipe pool at 4 for CPU, skips thread pool for sessions that don't need MediaPipe. |
| `core/proctor_session.py` | Added `debug_mode`, `_last_mp_result`, `_last_detections`, `set_debug()`, `get_debug_frame()`, `_draw_risk_panel()`, `needs_mediapipe()`, `_NULL_HEAD`. |
| `core/metrics.py` | NEW — in-process metrics singleton with background resource sampler. |
| `utils/logging_config.py` | NEW — structured JSON logging with rotation. |
| `detectors/head_pose_detector.py` | Removed `if not self.DEBUG: return` guard from `draw_debug()`. |
| `settings/scoring.py` | Removed grace period from tab_switch — all occurrences add score immediately. |
| `core/risk_engine.py` | `handle_tab_switch()` adds fixed score on every occurrence including first. |
| `frontend/app/admin/monitor/page.tsx` | NEW — system monitor page. |
| `frontend/app/candidate/page.tsx` | Added window.blur handler, copy/paste prevention, demo exam card. |
| `frontend/lib/api.ts` | Added `getMetrics()`, `getSystemReport()`, `toggleDebug()`, `sessionLog()`. |
| `frontend/lib/types.ts` | Added `MetricsSnapshot`, `SystemReport`, `EndpointMetric` interfaces. |

---

## 12. Dependencies

### Backend (Python)
```
fastapi, uvicorn, aiortc, av
ultralytics      ← YOLO (requires torch)
mediapipe        ← FaceMesh, head pose, lip detection
torch            ← for YOLO + VRAM checks
cv2 (opencv-python)
numpy
psutil
pynvml           ← optional, for GPU utilization %
```

### Frontend (Node.js)
```
next 14, react 18, typescript
tailwindcss
```

---

## 13. Running the Project

### Backend
```bash
cd Proctor-webRTC
python main.py                          # CPU, port 8000
python main.py --device cuda --half    # GPU FP16
python main.py --device cpu --port 9000
```

### Frontend
```bash
cd frontend
npm install
npm run dev     # http://localhost:3000
```

### URLs
- Admin dashboard: `http://localhost:3000/admin`
- System monitor: `http://localhost:3000/admin/monitor`
- Candidate page: `http://localhost:3000/candidate`
- Backend health: `http://localhost:8000/metrics`

---

*Last updated: 2026-03-17. CPU build complete and stable. GPU build is next.*
