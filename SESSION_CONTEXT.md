# ProctorPod — Session Context & Handoff Document

> **Read this at the start of every new session.** Also read `CLAUDE.md` for full system architecture.
> This file covers: what has been built, what has been deployed, current status, and exactly what to do next.

---

## 1. What This Project Is

An **AI-powered online exam proctoring system**:
- **Backend**: FastAPI + WebRTC (aiortc) running on a RunPod GPU server (RTX 3090)
- **Frontend**: Next.js 14 — admin dashboard + candidate exam page
- **AI pipeline**: YOLOv8 (object detection) + MediaPipe FaceMesh (head pose, lip movement) + Silero-VAD (audio)
- **Deployed on**: RunPod cloud GPU (Docker container, pushed to Docker Hub)

---

## 2. Current Infrastructure State

### RunPod Pod
- **Pod name**: ProctorPod
- **GPU**: RTX 3090 (Ampere sm_86, 24 GB VRAM) — $0.34/hr
- **Pod ID prefix**: `anj2lkrpqd8x7c`
- **Docker image**: `krushang08/proctor-webrtc:latest` on Docker Hub
- **SSH access**:
  - Via RunPod relay: `ssh anj2lkrpqd8x7c-64410b38@ssh.runpod.io -i ~/.ssh/id_ed25519`
  - Via direct TCP: `ssh root@213.192.2.72 -p 40086 -i ~/.ssh/id_ed25519`
- **HTTP Proxy URL**: `https://anj2lkrpqd8x7c-8000.proxy.runpod.net`
  - **WARNING**: This URL is blocked on office network by FortiGuard IPS (category: Proxy Avoidance)
  - Works fine on home network or mobile hotspot
- **Pod was stopped** before leaving office — **you need to start it again** from RunPod dashboard

### Docker Image
- Registry: Docker Hub → `krushang08/proctor-webrtc:latest`
- Last pushed: after RTX 3090 switch (stable build, no known startup errors)
- Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- PyTorch: `cu124` wheels (matches CUDA 12.4)
- CMD: `python main.py --device auto --half --warmup 3`

### Frontend
- Running locally on dev machine: `npm run dev` → `http://localhost:3000`
- `.env.local` currently set to RunPod proxy URL:
  ```
  NEXT_PUBLIC_BACKEND_URL=https://anj2lkrpqd8x7c-8000.proxy.runpod.net
  ```
- **CSS/Tailwind**: Fixed — `postcss.config.mjs` uses `@tailwindcss/postcss` with explicit `base: __dirname`

---

## 3. What Has Been Completed

### Backend (fully working)
- [x] FastAPI + aiortc WebRTC signaling (`POST /offer`)
- [x] YOLOv8 batch inference (GPU, FP16, warmup, auto device selection)
- [x] MediaPipe FaceMesh (head pose, gaze, lip movement, blink detection)
- [x] Silero-VAD audio monitoring via WebRTC push mode (`start_webrtc_mode`)
- [x] Risk scoring engine (fixed + decaying buckets, state machine)
- [x] Alert engine with cooldowns
- [x] SSE streaming (`GET /stream/{pc_id}`) for live alerts to admin
- [x] Debug overlay toggle (`POST /debug/{pc_id}`) with CV2 annotations
- [x] Tab switch detection + termination at 3 switches
- [x] Metrics endpoint (`GET /metrics`) + System report (`GET /system/report`)
- [x] Proof capture (JPEG + WAV on alert)
- [x] Structured JSON logging with rotation
- [x] Docker image built and pushed to Docker Hub
- [x] Running on RTX 3090 with CUDA, FP16, warmup — confirmed in logs

### Frontend (fully working)
- [x] Admin dashboard (`/admin`) — session cards with live risk
- [x] Session detail (`/admin/session/[pc_id]`) — snapshot, debug toggle, SSE alerts
- [x] System monitor (`/admin/monitor`) — live metrics + system report
- [x] Candidate page (`/candidate`) — WebRTC webcam/mic, tab switch detection, copy-paste disabled
- [x] Tailwind CSS rendering fixed
- [x] `lib/api.ts` — all API calls typed
- [x] `lib/types.ts` — all TypeScript interfaces

### Deployment
- [x] Docker Hub image pushed
- [x] RunPod pod deployed with RTX 3090
- [x] Backend confirmed running (logs showed `ObjectDetector ready device=cuda half=True compute=8.6`)

---

## 4. The ONE Remaining Problem — Network Access

### What happened
The office network has **FortiGuard IPS** which blocks `*.proxy.runpod.net` (category: "Proxy Avoidance").
This blocked:
1. Browser access to `https://anj2lkrpqd8x7c-8000.proxy.runpod.net` → 403 Forbidden (from FortiGuard, not our code)
2. SSH via `ssh.runpod.io` → connection hangs (also blocked)

### What to do at home (this evening)

**Step 1**: Start the RunPod pod from dashboard (it was stopped)

**Step 2**: Get updated connection details from RunPod "Connect" tab (IP/port may change on restart)

**Step 3**: Test SSH access — try both methods:
```bash
# Method A (RunPod relay)
ssh anj2lkrpqd8x7c-64410b38@ssh.runpod.io -i ~/.ssh/id_ed25519

# Method B (direct TCP — IP/port from RunPod Connect tab)
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

**Step 4**: Verify server is running inside pod:
```bash
ps aux | grep python
tail -50 /app/logs/app.log
curl -s http://localhost:8000/sessions
```

If server is not running (pod restart kills the container process), restart it:
```bash
cd /app
nohup python main.py --device auto --half --warmup 3 > /app/logs/stdout.log 2>&1 &
```

**Step 5**: Update frontend `.env.local` with the current proxy URL from RunPod Connect tab:
```
NEXT_PUBLIC_BACKEND_URL=https://<new-pod-id>-8000.proxy.runpod.net
```

**Step 6**: Test from browser — open the proxy URL directly in browser first, accept the SSL warning (self-signed cert), then try the frontend.

**Step 7**: Test single user end-to-end:
1. Open `http://localhost:3000/candidate` — should ask for camera/mic permission
2. Click "Start Exam" — should connect via WebRTC to backend
3. Open `http://localhost:3000/admin` — should see a session card appear
4. Click on session — should see live snapshot and risk score

---

## 5. WebRTC + HTTPS Note

Browsers require HTTPS (or localhost) to access camera and microphone via `getUserMedia()`.

- `https://anj2lkrpqd8x7c-8000.proxy.runpod.net` → HTTPS ✓ (RunPod's proxy provides TLS)
- `http://IP:PORT` direct → HTTP only → getUserMedia blocked
- **SSH tunnel workaround** (if proxy is blocked on your network):
  ```bash
  ssh -N -L 8000:localhost:8000 root@<IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
  ```
  Then use `http://localhost:8000` as backend URL (localhost is trusted for getUserMedia).

---

## 6. After Single User Test Passes — Load Test

The `load_test/` directory has everything needed:

### Files
- `load_test/client.py` — single WebRTC client that streams a video file
- `load_test/runner.py` — spawns N concurrent clients
- `load_test/requirements.txt` — `aiortc, aiohttp, av, opencv-python, numpy, sounddevice`

### Setup
```bash
cd load_test
pip install -r requirements.txt
```

### You still need a test video
Record yourself with webcam (5 min). The `record.py` script in `load_test/` is ready:
```bash
cd load_test

# List available devices first
python record.py --list-devices

# Record (auto-detects webcam mic, 5 minutes)
python record.py --duration 300 --output test_video.mp4

# If auto-detect picks wrong mic, specify manually
python record.py --duration 300 --output test_video.mp4 --mic 2
```

**Known fixes already in record.py**:
- Vertical flip fixed (`cv2.flip(frame, 0)`)
- Frame padding for correct duration
- sounddevice audio + ffmpeg mux for audio
- `--list-devices` flag to find webcam mic
- `--mic` flag to override device index

### Run load test (after single user confirmed working)
```bash
# Test with 5 users first
python runner.py --url https://<pod-url>/offer --video test_video.mp4 --clients 5

# Scale to 25 users
python runner.py --url https://<pod-url>/offer --video test_video.mp4 --clients 25
```

Monitor via admin dashboard or `GET /metrics` while test runs.

---

## 7. SSH Quick-Deploy (No Docker Rebuild Needed for Code Changes)

For any backend code changes (no new dependencies), use SSH instead of rebuilding Docker:

```bash
# From project root on your laptop
cd Proctor-webRTC

# Copy changed files to pod
scp server.py root@<IP>:/app/server.py -p <SSH_PORT> -i ~/.ssh/id_ed25519
scp core/proctor_session.py root@<IP>:/app/core/proctor_session.py -p <SSH_PORT> -i ~/.ssh/id_ed25519

# SSH in and restart
ssh root@<IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
pkill -f "python main.py"
cd /app && nohup python main.py --device auto --half --warmup 3 > logs/stdout.log 2>&1 &
tail -f logs/app.log
```

Only rebuild Docker when:
- `requirements.txt` changes (new Python package)
- `Dockerfile` changes
- Model weights change (`finalBestV5.pt`)

---

## 8. Critical Bug History (Do Not Re-Introduce)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `ModuleNotFoundError: pyaudio` on server startup | Top-level `import pyaudio` in `audio_monitor.py` | Made it lazy import inside `_run()` only; server uses `start_webrtc_mode()` which never needs pyaudio |
| `torchaudio CUDA version mismatch` | torchaudio installed from default PyPI (cu121) while torch was cu124 | All three (`torch torchvision torchaudio`) installed together with `--index-url https://download.pytorch.org/whl/cu124` |
| `CUDA error: no kernel image` with `--half` | RTX PRO 4500 is Blackwell (sm_120), PyTorch 2.6 has no FP16 kernels for it | Switched GPU to RTX 3090 (Ampere sm_86) — fully supported |
| YOLO bounding boxes in wrong position | Pre-resizing frames AND passing `imgsz` at runtime caused double coordinate transform | Never pre-resize; never pass `imgsz` at runtime; let YOLO handle letterboxing internally |
| Backend freeze on startup | `model.half()` + warmup runs in `__init__()` blocking asyncio event loop | Warmup moved to `_schedule_warmup()` daemon thread; defaults are `half=False, warmup=0` (CLI flags override) |
| Tab switch not terminating at 3 | Browser suspended the `fetch` when tab went hidden | `fetch(..., keepalive: true)` in candidate frontend |
| Alert flooding after termination | Tick loop kept processing after session end | `if self.risk.terminated: return` as first line of `session.update()` |

---

## 9. GPU Switch History (Important Context)

Originally selected **RTX PRO 4500** (Blackwell, sm_120) but had to switch to **RTX 3090** because:
- Blackwell (sm_120) requires PyTorch 2.8+ compiled for CUDA 12.8+
- PyTorch 2.6 (latest at time of build) has NO compiled kernels for sm_120
- Symptom: `CUDA error: no kernel image is available for execution on the device` on `--half`
- RTX 3090 = Ampere (sm_86), fully supported by PyTorch cu124, FP16 works perfectly

**RTX 3090 advantages for this workload**:
- 32 vCPUs (important — MediaPipe runs on CPU, needs parallelism)
- 24 GB VRAM (YOLO only uses ~1-2 GB, leaves headroom)
- $0.34/hr on RunPod
- FP16 fully supported → ~2x YOLO throughput

---

## 10. File-Level Changes Made During This Session

| File | Change |
|------|--------|
| `Proctor-webRTC/Dockerfile` | Base → CUDA 12.4.1, PyTorch cu124, CMD with --half --warmup |
| `Proctor-webRTC/requirements.txt` | Removed pyaudio, added silero-vad, nvidia-ml-py, av pin |
| `Proctor-webRTC/core/audio_monitor.py` | Removed top-level `import pyaudio`, made it lazy inside `_run()` |
| `Proctor-webRTC/config.py` | MAX_SESSIONS=40, YOLO_HALF=True, YOLO_WARMUP_FRAMES=3, YOLO_MIN_VRAM_GB=2.0 |
| `frontend/.env.local` | NEXT_PUBLIC_BACKEND_URL set to RunPod proxy URL |
| `frontend/postcss.config.mjs` | `@tailwindcss/postcss` with `base: __dirname` fix |
| `frontend/app/globals.css` | `@import "tailwindcss"` + `@source ".."` |
| `load_test/record.py` | Vertical flip, frame padding, sounddevice audio, --list-devices, --mic flags |

---

## 11. Immediate Next Steps (In Order)

1. **Start RunPod pod** from dashboard
2. **Get new connection details** from Connect tab (IP may change)
3. **SSH in** and verify server running
4. **Update `.env.local`** with current proxy URL
5. **Test single user** (candidate page → admin page)
6. **Record test video** with `record.py` (5 min)
7. **Run load test** with 5 → 25 clients
8. **Monitor** via `/metrics` endpoint and admin dashboard
9. **Tune** `MAX_SESSIONS` in `config.py` based on actual GPU profiling

---

## 12. Useful Commands Reference

```bash
# Start frontend
cd frontend && npm run dev

# Start backend locally (CPU)
cd Proctor-webRTC && python main.py

# Start backend locally (GPU)
cd Proctor-webRTC && python main.py --device cuda --half --warmup 3

# Build and push Docker image
cd Proctor-webRTC
docker build -t krushang08/proctor-webrtc:latest .
docker push krushang08/proctor-webrtc:latest

# SSH port tunnel (bypass proxy for localhost access)
ssh -N -L 8000:localhost:8000 root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Backend health check
curl http://localhost:8000/sessions
curl http://localhost:8000/metrics

# Record test video
cd load_test && python record.py --list-devices
cd load_test && python record.py --duration 300 --output test_video.mp4

# Run load test
cd load_test && python runner.py --url https://<pod-url>/offer --video test_video.mp4 --clients 5
```

---

*Last updated: 2026-03-18. Backend deployed on RTX 3090, frontend working locally. Blocked on office network FortiGuard — resume testing from home.*
