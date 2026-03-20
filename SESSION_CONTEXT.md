# ProctorPod — Session Context & Handoff Document

> **Read this at the start of every new session.** Also read `CLAUDE.md` for full system architecture.
> This file is append-versioned — past decisions are preserved so you can understand *why* things are the way they are.
> Last updated: 2026-03-20

---

## 1. What This Project Is

An **AI-powered online exam proctoring system**:
- **Backend**: FastAPI + WebRTC (aiortc), Python — lives in `Proctor-webRTC/`
- **Frontend**: Next.js 14 admin dashboard + candidate exam page — lives in `frontend/`
- **AI pipeline**: YOLOv8 (object detection) + MediaPipe FaceMesh (head pose, lip, blink) + Silero-VAD (audio)
- **Load test tools**: `load_test/` — `runner.py`, `client.py`, `benchmark.py`, `report.py`

---

## 2. Infrastructure — Current State (AWS EC2)

### Why we left RunPod (historical)
1. Office network FortiGuard IPS blocked `*.proxy.runpod.net` (category: "Proxy Avoidance") → 403 on every request
2. RunPod containers block all UDP ports → WebRTC media streams fail (state→failed)
3. Container creation kept failing with `context deadline exceeded` / `exit status 1`
4. SSH relay `ssh.runpod.io` also blocked by FortiGuard

### Current: AWS EC2 g4dn.xlarge (Mumbai — ap-south-1) ✅ RUNNING

- **Instance name**: `proctor-backend`
- **Region**: `ap-south-1` (Mumbai)
- **Instance type**: `g4dn.xlarge` — 4 vCPU, 16 GB RAM, 1× NVIDIA T4 GPU (16 GB VRAM)
- **AMI**: `Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)` — ami-076dd1c646bcc16c4
- **SSH username**: `ubuntu`
- **Key pair**: `proctor-key` (.pem file — on local machine)
- **Elastic IP**: `13.201.166.165` (permanent — never changes on stop/start)
- **Security group** `proctor-sg`:
  - TCP 22 (SSH), TCP 8000 (FastAPI), UDP 10000–60000 (WebRTC media)
- **Storage**: 50 GB gp3
- **Pricing**: ~$0.586/hr On-Demand

### Docker image
- **Registry**: `krushangshahdrc18/proctor-backend:latest`
- **Base**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- **PyTorch**: cu124 wheels
- **CMD**: `python main.py --device auto --half --warmup 3`
- **Key pinned versions**: `mediapipe==0.10.11`, `protobuf==3.20.3`

### Run command (always use --network host for WebRTC UDP)
```bash
docker run -d --name proctor --gpus all --network host --restart unless-stopped krushangshahdrc18/proctor-backend:latest
```

> **Why `--network host`**: With Docker bridge networking, STUN returns the container's internal IP (172.x.x.x) as the ICE candidate. The browser can't reach that → WebRTC stays `connecting` forever. Host networking makes STUN see the real EC2 public IP → ICE succeeds. Note: drop `-p 8000:8000` when using host networking (port mapping is ignored).

### View logs
```bash
docker logs -f proctor              # follow live
docker logs --tail 100 -f proctor   # last 100 lines + follow
docker logs --since 5m proctor      # last 5 minutes
```

### Frontend config
```
# frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://13.201.166.165:8000
```

---

## 3. Quick Reference Commands

```bash
# SSH into EC2
ssh -i /path/to/proctor-key.pem ubuntu@13.201.166.165

# GPU check
nvidia-smi

# Docker management
docker ps
docker logs -f proctor
docker restart proctor
docker exec -it proctor bash

# Health checks
curl http://13.201.166.165:8000/metrics
curl http://13.201.166.165:8000/sessions
curl http://13.201.166.165:8000/system/report

# SSH tunnel for HTTPS workaround (getUserMedia)
ssh -N -L 8000:localhost:8000 -i proctor-key.pem ubuntu@13.201.166.165
# Then set: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Frontend dev
cd frontend && npm run dev   # http://localhost:3000

# Build + push Docker image
cd Proctor-webRTC
docker build -t krushangshahdrc18/proctor-backend:latest .
docker push krushangshahdrc18/proctor-backend:latest

# Deploy on EC2 (single line)
docker pull krushangshahdrc18/proctor-backend:latest && docker stop proctor && docker rm proctor && docker run -d --name proctor --gpus all --network host --restart unless-stopped krushangshahdrc18/proctor-backend:latest
```

---

## 4. Current Configuration (config.py)

| Key | Value | Why |
|-----|-------|-----|
| `MAX_SESSIONS` | 5 | Benchmarked safe limit on g4dn.xlarge at full detection quality |
| `TICK_RATE` | 10 Hz | Duration gates are 1.5–2 s; 10 Hz gives 15–20 samples per window |
| `YOLO_DEVICE` | `"auto"` | Uses CUDA if available, falls back to CPU |
| `YOLO_HALF` | `True` | FP16 on T4: ~2× throughput, negligible quality loss |
| `YOLO_WARMUP_FRAMES` | 3 | Pre-compile CUDA kernels at startup |
| `YOLO_IMGSZ` | 640 | Do NOT reduce — 320 loses earbud detection (small object) |
| `MEDIAPIPE_STRIDE` | 3 | Fresh FaceMesh every 3rd tick per session; reuse last result otherwise |

---

## 5. Architecture — Inference Pipeline

```
Tick loop (10 Hz target)
├── asyncio.gather() — YOLO + MediaPipe run CONCURRENTLY
│   ├── YOLO batch → GPU  (all N frames in one forward pass)
│   └── MediaPipe thread pool (stride=3: only ceil(N/3) sessions per tick)
│       └── session.run_mediapipe(frame) → FaceMesh → HeadPose → Lip
└── session.update() for each session — sequential, fast CPU
    ├── ObjectTemporalTracker (time-based vote window)
    ├── HeadTracker (duration gates)
    ├── RiskEngine (score accumulation)
    └── SSE push to admin frontend
```

**Why YOLO + MP concurrent**: YOLO is GPU-bound. While the GPU runs inference the CPU is free to run MediaPipe. Using `asyncio.gather(yolo_task, mp_gather)` makes wall-clock cost = `max(YOLO_ms, MP_ms)` instead of sum.

**Why stride=3**: At 5 users, serial MP = 5×15ms = 75ms per tick. With stride=3: ceil(5/3)=2 sessions per tick = 30ms. Since YOLO takes ~60ms, MP is hidden behind YOLO in the overlap.

---

## 6. Change Log (Versioned)

### v1.0 — Initial CPU Build (Local Machine)
**Date**: Before 2026-03-19
**What existed**: Basic working proctoring system, CPU-only, `MAX_SESSIONS=3`.
- YOLO + MediaPipe ran sequentially in the tick loop
- MediaPipe called in `ThreadPoolExecutor`, but run sequentially per tick (one await after another)
- Profiled: 1 user = 94ms YOLO (CPU) + 40ms MP = 134ms tick → 7.5 Hz

---

### v1.1 — RunPod GPU Attempt (Failed — abandoned)
**Date**: ~2026-03-18
**What was tried**: Deployed to RunPod GPU container.
**Why failed**:
- Office FortiGuard IPS blocked `*.proxy.runpod.net` → 403 on all requests
- RunPod blocked all UDP → WebRTC `state→failed`
- SSH relay also blocked
**Decision**: Migrated to AWS EC2 g4dn.xlarge (Mumbai) with open UDP 10000–60000.

---

### v1.2 — AWS EC2 GPU Build + WebRTC Stability Fixes
**Date**: 2026-03-19
**Changes**:
- Migrated to `g4dn.xlarge` (T4 GPU, 4 vCPU, 16 GB RAM)
- `YOLO_HALF=True`, `YOLO_WARMUP_FRAMES=3`, `YOLO_DEVICE="auto"`
- Added `argparse` CLI (`--device`, `--half`, `--warmup`, `--port`) to `main.py`
- Added `uvloop` (libuv asyncio event loop — lower per-coroutine overhead)
- Added `RequestMetricsMiddleware`, `GET /metrics`, `GET /system/report`
- Added `MEDIAPIPE_STRIDE=3` + `YOLO_IMGSZ=640` to `config.py`
- Coordinator: stagger-rotates which sessions get fresh MP each tick
- Coordinator: YOLO + MP submitted concurrently via `asyncio.gather()`

**WebRTC freeze bugs fixed**:

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Frame freeze after 2–3 min | `asyncio.wait_for` cancelled `track.recv()` mid-flight → corrupted aiortc jitter buffer | Replaced with `asyncio.wait({task}, timeout=N)` — never cancels tasks |
| `ValueError: No decoder found for MIME type 'video/rtx'` | aiortc negotiated RTX retransmission alongside VP8; browser sent RTX packets → `decoder_worker` thread crashed → `recv()` blocked forever | Added `_strip_rtx_from_sdp()` — removes all RTX payload types from offer SDP before `setRemoteDescription` |
| Frames not decoded | `frame.to_ndarray(format="bgr24")` raised on corrupt frames | Wrapped in try-except, skip frame on error |

**Confirmed**: 5-minute streaming stable, no freezes.

---

### v1.3 — ProcessPoolExecutor Experiment (Reverted)
**Date**: 2026-03-20
**What was tried**: Replace `ThreadPoolExecutor` for MediaPipe with `ProcessPoolExecutor` (spawn context) to bypass the GIL and get true parallel FaceMesh execution.

**Architecture**:
- `facemesh_worker.py` (top-level): per-process FaceMesh singleton, `static_image_mode=True`, returns float32 landmark bytes via IPC
- Frames downscaled to 320×240 before IPC (4× less data)
- `_FakeLandmark` duck-type class in session to deserialize landmark bytes
- `run_headpose_lip(frame, lm_bytes)` in session for phase-2 HeadPose+Lip
- YOLO (GPU) and FaceMesh (process pool) ran concurrently via `asyncio.gather()`

**Why reverted**:
- IPC overhead (pickle/unpickle frame bytes + landmark bytes per session per tick) added unpredictable latency spikes
- Occasional worker stalls → `_tick()` caught `Exception` and returned early → YOLO results silently dropped for that tick
- Dropped ticks broke `ObjectTemporalTracker` vote accumulation → temporal vote windows never filled → earbud/phone detections failed intermittently for all users
- ProcessPool benefit only materialises at 10+ concurrent users; at target of 5 users it was net negative
- For 5 users/stride=3: only 2 serial MP calls per tick = 30ms → already hidden behind YOLO's ~60ms GPU time

**Decision**: Reverted to `ThreadPoolExecutor`. Kept the `asyncio.gather()` YOLO+MP overlap — that's a real win regardless. `facemesh_worker.py` kept in codebase for potential future use at higher user counts.

---

### v1.4 — MAX_SESSIONS=5 + Detection Fix for Later-Joining Users
**Date**: 2026-03-20

#### Change 1: MAX_SESSIONS reduced to 5
**Why**: Benchmarked 2, 5, 6, 7 users. Quality and tick rate hold well at 5. Above that, tick latency climbs and detection reliability degrades. 5 is the confirmed safe limit on g4dn.xlarge.

#### Change 2: `_tick_fps` initial value bug fix
**Bug**: Users 3 and 4 (later-joining sessions) had severely degraded detection — earbuds, phones not triggering.

**Root cause**: `ObjectTemporalTracker.update()` computes:
```python
min_votes = max(MIN_VOTES_FLOOR, int(fps × window_s × ratio))
```
This uses `session._tick_fps` which is an EMA of the actual coordinator tick rate. `_tick_fps` was **initialised at 15.0** (wrong — that's the WebRTC camera fps, not the coordinator rate).

Actual coordinator rate on 4-user load: ~8 Hz.

For earbud (`ratio = EARBUD_MIN_VOTES/OBJECT_WINDOW = 9/15 = 0.60`, `window_s = 1.0`):
- With `_tick_fps=15.0`: `min_votes = int(15 × 1.0 × 0.60) = 9` required in 1 second
- But only ~8 ticks/second available → **9 > 8 → physically impossible to confirm**

Users 1 & 2 worked fine because their `_tick_fps` had already converged to ~8 Hz over many ticks (EMA α=0.1 took ~25 ticks = 2.5 seconds to converge). Users 3 & 4 were still at 15.0.

**Fix**:
1. `proctor_session.py`: `self._tick_fps = 15.0` → `self._tick_fps = 10.0` (coordinator target rate)
2. EMA alpha: `0.9/0.1` → `0.75/0.25` — converges to true rate in ~5 ticks (0.5 s) instead of ~25 ticks (2.5 s)

With `_tick_fps=10.0`: `min_votes = int(10 × 1.0 × 0.60) = 6` — achievable at 8 Hz (needs 75% positive detections). ✅

**Files changed**:

| File | Change |
|------|--------|
| `config.py` | `MAX_SESSIONS = 5` |
| `core/proctor_session.py` | `_tick_fps` init: `15.0 → 10.0`; EMA alpha: `0.1 → 0.25` |
| `core/proctor_coordinator.py` | Reverted ProcessPool → ThreadPool; kept YOLO+MP `asyncio.gather()` overlap; kept stride=3 |

---

## 7. All Known Bugs Fixed (Do Not Re-Introduce)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `mediapipe AttributeError: no attribute 'solutions'` | pip installed mediapipe 0.11+ | Pin: `mediapipe==0.10.11` + `protobuf==3.20.3` |
| `ModuleNotFoundError: pyaudio` on startup | Top-level `import pyaudio` | Lazy import inside `_run()` only |
| `torchaudio CUDA version mismatch` | torchaudio from PyPI cu121 vs torch cu124 | Install all three together with `--index-url .../cu124` |
| YOLO bboxes in wrong position | Pre-resize frames + `imgsz` at runtime = double coordinate transform | Never pre-resize; never pass `imgsz` at inference time; set at load via `model.overrides` |
| Backend freeze on startup | `model.half()` + warmup in `__init__()` blocked asyncio event loop | Warmup moved to `_schedule_warmup()` daemon thread |
| Tab switch not terminating at count 3 | Browser suspended fetch when tab hidden | `fetch(..., keepalive: true)` in candidate page |
| Alert flooding after termination | Tick loop kept processing terminated session | `if self.risk.terminated: return` in `session.update()` |
| ERR_EMPTY_RESPONSE on /offer | SSL cert files present → server switched to HTTPS; frontend using HTTP | `mv ~/key.pem ~/key.pem.bak` disables HTTPS mode |
| WebRTC `state→connecting` forever on EC2 Docker | Bridge networking → STUN returns container IP → browser can't reach it | `--network host` in docker run command |
| Frame freeze after 2–3 min | `asyncio.wait_for` cancelled `track.recv()` → corrupted jitter buffer | Use `asyncio.wait({task}, timeout=N)` — never cancels tasks on timeout |
| `video/rtx` decoder crash → freeze | aiortc negotiated RTX; browser sent RTX packets → `decoder_worker` crashed | `_strip_rtx_from_sdp()` removes RTX PTs from offer SDP before `setRemoteDescription` |
| Users 3+ miss detections (earbud/phone) | `_tick_fps` init at 15.0; `min_votes = int(15×1.0×0.60) = 9` but only 8 ticks/s available | Init `_tick_fps = 10.0`; increase EMA alpha to 0.25 for fast convergence |
| ProcessPool dropped YOLO ticks | Worker stalls → `_tick()` returned early → temporal vote windows never filled | Reverted to ThreadPoolExecutor for 5-user target |

---

## 8. Scaling Reference

### Benchmarked results (g4dn.xlarge, current build)
| Users | Tick rate | Detection quality | Verdict |
|-------|-----------|-------------------|---------|
| 1–2 | ~10 Hz | Excellent | ✅ |
| 3–5 | ~8 Hz | Good | ✅ Safe limit |
| 6–7 | ~6–7 Hz | Degraded | ⚠️ Misses fast events |
| 10+ | ~4–5 Hz | Poor | ❌ |

### Running 2 containers on 1 instance (8 users total)
Both containers use `--network host`. aiortc assigns random UDP ports per connection — no collision.
Container 1 → port 8000 (default CMD)
Container 2 → port 8001:
```bash
docker run -d --name proctor2 --gpus all --network host --restart unless-stopped krushangshahdrc18/proctor-backend:latest python main.py --device auto --half --warmup 3 --port 8001
```
GPU is time-shared. YOLO latency roughly doubles per container (~60ms → ~120ms). Tick rate ~8 Hz still sufficient. Watch CPU — 4 vCPUs for 2 containers is tight.

### EC2 instance upgrade path
| Instance | vCPU | RAM | GPU | Est. safe users | On-Demand/hr |
|----------|------|-----|-----|-----------------|--------------|
| g4dn.xlarge | 4 | 16 GB | T4 16GB | 5 (current) | $0.586 |
| g4dn.2xlarge | 8 | 32 GB | T4 16GB | 10–12 | ~$0.752 |
| g4dn.4xlarge | 16 | 64 GB | T4 16GB | 20–25 | ~$1.204 |

Upgrade trigger: if tick latency at 5 users exceeds 150ms consistently.

---

## 9. HTTPS for Camera Access

Browsers block `getUserMedia()` on plain HTTP except `localhost`.

**Option A — SSH Tunnel (easiest for testing)**
```bash
ssh -N -L 8000:localhost:8000 -i proctor-key.pem ubuntu@13.201.166.165
# Set: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

**Option B — Self-signed cert + nginx**
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/proctor.key -out /etc/ssl/proctor.crt \
  -subj "/CN=13.201.166.165"
# nginx reverse proxy: 443 → localhost:8000
# Set: NEXT_PUBLIC_BACKEND_URL=https://13.201.166.165
```

**Option C — ngrok (quickest demo)**
```bash
ngrok http 8000
# Use the https://xxxx.ngrok.io URL
```

---

## 10. Features Built (Complete List)

### Backend
- [x] FastAPI + aiortc WebRTC signaling (`POST /offer`)
- [x] YOLOv8 batch inference (GPU FP16, warmup, auto device selection)
- [x] MediaPipe FaceMesh — head pose, gaze, lip movement, blink detection
- [x] Silero-VAD audio monitoring via WebRTC PCM push mode
- [x] Risk scoring (fixed + decaying buckets, NORMAL→WARNING→HIGH_RISK→TERMINATED)
- [x] Alert engine with cooldowns + SSE streaming to admin
- [x] Debug overlay toggle (CV2 annotations on snapshot)
- [x] Tab switch detection + auto-terminate at 3 switches
- [x] 5-minute hard exam time limit
- [x] `GET /metrics` + `GET /system/report`
- [x] Proof capture (JPEG per alert, WAV for audio alerts)
- [x] Structured JSON logging with rotation
- [x] Trickle ICE — offer sent immediately, ICE candidates trickle via `POST /ice-candidate/{pc_id}`
- [x] 8-second disconnect grace on WebRTC `disconnected` state
- [x] MediaPipe + audio VAD latency tracking in metrics
- [x] RTX SDP strip — prevents aiortc `video/rtx` decoder crash
- [x] VP8 codec preference forcing via `setCodecPreferences`
- [x] YOLO + MediaPipe concurrent via `asyncio.gather()` (overlap GPU + CPU work)
- [x] MediaPipe stride=3 — reduces MP CPU load 3× with no quality loss at 10 Hz
- [x] `_tick_fps` correctly initialised at 10.0 — all users detect equally from first tick

### Frontend
- [x] Admin dashboard (`/admin`) — live session cards
- [x] Session detail (`/admin/session/[pc_id]`) — snapshot, debug toggle, SSE alert log
- [x] System monitor (`/admin/monitor`) — live metrics + system report
- [x] Candidate page (`/candidate`) — WebRTC, tab switch, window blur, copy-paste disabled
- [x] Trickle ICE — `onicecandidate` sends to `/ice-candidate/{pc_id}`
- [x] 8s disconnect grace — `onconnectionstatechange` with `setTimeout`

### Load Test Tools (`load_test/`)
- [x] `client.py` — single WebRTC client streaming a video file
- [x] `runner.py` — N concurrent clients with live metrics dashboard
- [x] `benchmark.py` — multi-level benchmark with HTML report (tick, YOLO, MP, VAD, CPU/RAM/GPU charts)
- [x] `report.py` — summary report generator

---

## 11. Next Optimisation (When Needed)

### TensorRT YOLO — biggest remaining gain (3–5× YOLO speedup)
Only needed if scaling beyond 5 users per container. T4-specific — must build on EC2.
```bash
docker exec -it proctor python -c "
from ultralytics import YOLO
model = YOLO('finalBestV5.pt')
model.export(format='engine', device=0, half=True, imgsz=640, batch=8)
print('done → finalBestV5.engine')
"
docker cp proctor:/app/finalBestV5.engine ./Proctor-webRTC/
```
Then: `YOLO_MODEL_PATH = os.path.join(BASE_DIR, "finalBestV5.engine")` in `config.py`.
Note: `.engine` is T4-specific. Re-export if switching instance type.

### ProcessPoolExecutor — revisit at 10+ users
The v1.3 approach was correct in principle but needs IPC reliability hardening:
- Add per-task timeout (`asyncio.wait_for` on individual FM futures, not the whole gather)
- Fallback: if worker result times out, reuse `_last_lm_bytes` silently
- Only worth doing if tick rate with ThreadPool + stride is insufficient for the target user count

---

## 12. Load Testing Workflow

```bash
cd load_test
# Add silent audio track to test video (required for VAD testing)
ffmpeg -i test_video.mp4 -f lavfi -i anullsrc=r=16000:cl=mono \
  -c:v copy -c:a aac -shortest -y test_video_with_audio.mp4

# Benchmark
python benchmark.py \
  --url http://13.201.166.165:8000 \
  --video test_video_with_audio.mp4 \
  --levels 1,5,10 \
  --warmup 30 \
  --steady 120

# Simple load test
python runner.py \
  --url http://13.201.166.165:8000 \
  --video test_video_with_audio.mp4 \
  --clients 5 \
  --duration 300
```

---

## 13. GitHub Repository

- **Repo**: https://github.com/krushangshah18/ProctoDeployBackendGPU.git

```bash
git add -p   # stage changes selectively
git commit -m "your message"
git push origin main
```

---

*Last updated: 2026-03-20. EC2 running and stable. MAX_SESSIONS=5, detection quality confirmed equal for all users. ProcessPool experiment reverted. `_tick_fps` bug fixed.*
