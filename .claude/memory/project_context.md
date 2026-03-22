---
name: ProctorPod project context
description: Architecture, tech stack, deployment, and evolution of the AI proctoring system
type: project
---

## What the project is

**ProctorPod** — AI-powered online exam proctoring system.

- **Backend**: FastAPI + aiortc WebRTC server (Python). Receives candidate webcam/audio via WebRTC, runs YOLOv8 batch GPU inference + MediaPipe FaceMesh + Silero VAD, emits alerts/scores over SSE.
- **Frontend**: Next.js 14 admin + candidate UI.
- **Deployment**: AWS EC2 g4dn.xlarge (Tesla T4 GPU). Two Docker containers: `proctor1:8000`, `proctor2:8001`. Max 3 active sessions per container (MediaPipe is the CPU bottleneck, not YOLO).
- **Docker image**: `krushangshahdrc18/proctor-backend:latest`

---

## What the build looked like BEFORE (CPU build)

- `YOLO_DEVICE = "auto"` — CPU/MPS/GPU auto-selection via `_resolve_device()`
- `MAX_SESSIONS = 3` (CPU limit)
- `BLINK_FRAMES = 2` — frame-count blink detection (unreliable at 3.3 Hz per session)
- `MIN_FACE_WIDTH = 80`, `MIN_FACE_HEIGHT = 95`
- `model.half()` and warmup ran synchronously in `ObjectDetector.__init__()` — blocked the event loop at startup
- `--device auto` was a CLI argument in `main.py`
- `CORS allow_origins = ["*"]`
- `MODEL_ENABLED = True` flag existed in `server.py` with dead `if MODEL_ENABLED:` guards everywhere
- `facemesh_worker.py` existed (abandoned ProcessPool two-phase FaceMesh approach)
- `local_runner.py`, `client.html`, `client.js`, `analysis.html`, `monitor.html` existed (dev test files)
- `_FakeLandmark` class and `run_headpose_lip()` method existed in `ProctorSession` (dead code from old approach)
- `HeadPoseDetector.__init__` read threshold values from module-level `from config import` — runtime_settings changes to face size / thresholds did NOT apply to new sessions
- Docker CMD: `python main.py --device auto --half --warmup 3`

---

## What changed and why (GPU production build)

### GPU-only, fail-fast
- `_resolve_device()` replaced with `_ensure_cuda()` — raises `RuntimeError` immediately if CUDA unavailable. No silent CPU degradation in production.
- `YOLO_DEVICE = "cuda"`, `YOLO_HALF = True`, `YOLO_WARMUP_FRAMES = 3`
- `--device` CLI arg removed from `main.py` (always CUDA)
- Docker CMD: `python main.py --half --warmup 3`
- **Why**: GPU-only EC2 deployment, no need for CPU fallback path.

### Dead code removal
- Deleted: `facemesh_worker.py`, `local_runner.py`, `client.html`, `client.js`, `analysis.html`, `monitor.html`
- Removed `MODEL_ENABLED` flag and all `if MODEL_ENABLED:` guards from `server.py`
- Removed `_FakeLandmark` + `run_headpose_lip()` from `ProctorSession`
- **Why**: Production readiness — no unused code paths.

### HeadPoseDetector constructor params fix
- `HeadPoseDetector.__init__` now accepts all threshold values as explicit constructor params (`min_face_width`, `min_face_height`, `look_away_yaw`, `look_down_pitch`, `look_up_pitch`, `gaze_left`, `gaze_right`, `ear_threshold`, `blink_min_duration_s`, `blink_max_duration_s`)
- `ProctorSession` passes all these from `session_cfg` which merges `coordinator.runtime_settings`
- Removed `_hpd_mod` module-patching hack from `server.py`
- **Why**: Runtime settings changes (via admin settings page) were not applying to new sessions because `HeadPoseDetector` read from module-level import bindings fixed at import time.

### Time-based blink detection
- Replaced `BLINK_FRAMES = 2` (frame-count) with time-based detection using `_blink_start` timestamp
- `BLINK_MIN_DURATION_S = 0.05` (50ms), `BLINK_MAX_DURATION_S = 0.40` (400ms)
- Both values tunable via admin settings page
- **Why**: At 10Hz/stride=3 each session gets MediaPipe at ~3.3Hz (~300ms per sample). `BLINK_FRAMES=2` meant ~600ms minimum — missed typical quick blinks (150–400ms).

### Face size defaults
- `MIN_FACE_WIDTH: 80 → 110`, `MIN_FACE_HEIGHT: 95 → 120`
- **Why**: Previous values were too permissive — partial face was not triggering when the candidate was far from camera.

### CORS hardening
- `allow_origins=["*"]` → reads from `CORS_ORIGINS` env var, defaults to `http://localhost:3000`
- `allow_methods=["GET", "POST"]` (was `["*"]`)
- **Why**: Production security — no need for wildcard origins.

### Warmup fix (startup freeze)
- Warmup moved to `_schedule_warmup()` daemon background thread
- **Why**: Sync warmup in `__init__` blocked the asyncio event loop during FastAPI lifespan startup.

### Reports management
- Backend: `GET /reports/meta` returns rich metadata (session_id, risk state, counts, disk size KB, proof count) in one call per backend
- Backend: `DELETE /report/{report_id}` removes report directory recursively (path-traversal safe)
- Frontend: `frontend/app/reports/page.tsx` — per-backend panels, table view, individual delete + bulk delete with confirm dialog
- Report viewer (`/report/[report_id]`) updated to accept `?backend=` query param for multi-instance support
- **Why**: Proof JPEG/WAV files were accumulating on disk and eating space.

---

## Config values (current production)

```python
TICK_RATE          = 10        # Hz
MAX_SESSIONS       = 5         # g4dn.xlarge: 3 active + 2 buffer
YOLO_DEVICE        = "cuda"
YOLO_HALF          = True
YOLO_WARMUP_FRAMES = 3
YOLO_MIN_VRAM_GB   = 2.0
YOLO_IMGSZ         = 640
MEDIAPIPE_STRIDE   = 3
MIN_FACE_WIDTH     = 110
MIN_FACE_HEIGHT    = 120
BLINK_MIN_DURATION_S = 0.05
BLINK_MAX_DURATION_S = 0.40
EAR_THRESHOLD      = 0.20
LOOK_AWAY_YAW      = 0.20
LOOK_DOWN_PITCH    = 0.13
LOOK_UP_PITCH      = -0.10
GAZE_LEFT          = -0.13
GAZE_RIGHT         = 0.13
```

---

## Docker run commands (current)

```bash
docker run -d --name proctor1 --gpus all --network host \
  --restart unless-stopped \
  krushangshahdrc18/proctor-backend:latest \
  python main.py --half --warmup 3 --port 8000

docker run -d --name proctor2 --gpus all --network host \
  --restart unless-stopped \
  krushangshahdrc18/proctor-backend:latest \
  python main.py --half --warmup 3 --port 8001
```

Note: `--device auto` was removed from these commands after `--device` CLI arg was removed.

---

## Known issues fixed

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Container crash-loop | Old docker run command still had `--device auto` | Removed `--device auto` from run commands |
| Warmup dtype warning | FP16 warmup with zeros triggers `Half != float` warning | Cosmetic — does not affect real inference |
| MIN_FACE_WIDTH not applying | HeadPoseDetector read module-level imports | Added explicit constructor params |
| Quick blinks not counted | BLINK_FRAMES=2 at 3.3Hz = 600ms min | Time-based detection with 50ms min |
| container name conflict | Old containers not removed before re-create | `docker rm -f proctor1 proctor2` first |
