# ProctorPod — Session Context & Handoff Document

> **Read this at the start of every new session.** Also read `CLAUDE.md` for full system architecture.
> This file covers: what has been built, current infrastructure state, exactly what to do next.
> Last updated: 2026-03-19 (office session — migrating from RunPod to AWS EC2)

---

## 1. What This Project Is

An **AI-powered online exam proctoring system**:
- **Backend**: FastAPI + WebRTC (aiortc), Python — lives in `Proctor-webRTC/`
- **Frontend**: Next.js 14 admin dashboard + candidate exam page — lives in `frontend/`
- **AI pipeline**: YOLOv8 (object detection) + MediaPipe FaceMesh (head pose, lip, blink) + Silero-VAD (audio)
- **Load test tools**: `load_test/` — `runner.py`, `client.py`, `benchmark.py`, `report.py`

---

## 2. Infrastructure — MIGRATED FROM RUNPOD TO AWS EC2

### Why we left RunPod
1. Office network FortiGuard IPS blocked `*.proxy.runpod.net` (category: "Proxy Avoidance") → 403 on every request
2. RunPod containers block all UDP ports → WebRTC media streams fail (state→failed)
3. Container creation kept failing with `context deadline exceeded` / `exit status 1` (physical machine failures)
4. SSH relay `ssh.runpod.io` also blocked by FortiGuard

### Target: AWS EC2 g4dn.xlarge (Mumbai — ap-south-1)

**Instance details (being set up right now):**
- **Instance name**: `proctor-backend`
- **Region**: `ap-south-1` (Mumbai)
- **Instance type**: `g4dn.xlarge` — 4 vCPU, 16 GB RAM, 1× NVIDIA T4 GPU (16 GB VRAM)
- **AMI**: `Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)` — ami-076dd1c646bcc16c4
  - Has CUDA pre-installed, Docker + nvidia-container-toolkit pre-configured
  - SSH username: `ubuntu`
- **Key pair**: `proctor-key` (.pem file — downloaded to local machine)
- **Security group**: `proctor-sg` with rules:
  - TCP 22 from 0.0.0.0/0 (SSH)
  - TCP 8000 from 0.0.0.0/0 (FastAPI backend)
  - UDP 10000–60000 from 0.0.0.0/0 (WebRTC media streams — critical, was blocked on RunPod)
- **Storage**: 50 GB gp3
- **Pricing**: ~$0.586/hr Ubuntu On-Demand (Mumbai)

### CURRENT BLOCKER — AWS vCPU Quota = 0

**The instance launch failed with:**
```
You have requested more vCPU capacity than your current vCPU limit of 0 allows
for the instance bucket that the specified instance type belongs to.
```

**What needs to be done RIGHT NOW:**
1. Go to AWS Console → **Service Quotas** → AWS Services → **Amazon EC2**
2. Search: `Running On-Demand G and VT instances`
3. Current value: **0** — click **Request increase at account level**
4. Request value: **4** (g4dn.xlarge needs exactly 4 vCPUs)
5. Description to write:
   ```
   Running a GPU-based AI inference server for exam proctoring.
   Requesting 4 vCPUs to launch one g4dn.xlarge instance.
   ```
6. Submit — approval is usually automatic or within 2–24 hours
7. You will get an email when approved

**After quota is approved:**
- Go back to EC2 → Launch Instance
- Everything is already configured (AMI, instance type, security group, storage)
- Just click Launch Instance again — all settings are preserved in the launch wizard
- Then immediately set up an **Elastic IP** (see Section 4 below)

---

## 3. Docker Image (Backend)

- **Registry**: Docker Hub → `krushang08/proctor-webrtc:latest`
- **Base**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- **PyTorch**: cu124 wheels
- **CMD**: `python main.py --device auto --half --warmup 3`
- **Key pinned versions**: `mediapipe==0.10.11`, `protobuf==3.20.3` (newer mediapipe breaks solutions API)
- **Last pushed**: Stable build — confirmed running on NVIDIA GPU with CUDA FP16

---

## 4. Step-by-Step: After Quota Approved — Launch & Setup

### Step 1 — Launch Instance
- EC2 Console → Launch Instance (settings already configured)
- Click Launch Instance
- Wait ~2 minutes for `running` state

### Step 2 — Elastic IP (Do immediately — critical)
Without Elastic IP, the instance gets a new public IP every stop/start, requiring `.env.local` update every time.
```
EC2 Console → Elastic IPs (left sidebar, Network & Security)
→ Allocate Elastic IP address → Allocate
→ Select new IP → Actions → Associate Elastic IP address
→ Choose instance: proctor-backend → Associate
```
**Note this IP down permanently** — it never changes.

### Step 3 — SSH into instance
```bash
chmod 400 /path/to/proctor-key.pem
ssh -i /path/to/proctor-key.pem ubuntu@<elastic-ip>
```

### Step 4 — Verify GPU
```bash
nvidia-smi
# Should show: Tesla T4, CUDA 12.x
```

### Step 5 — Verify Docker GPU access
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# Should show T4 inside Docker
```

### Step 6 — Pull and run Docker container
```bash
docker pull krushang08/proctor-webrtc:latest

docker run -d \
  --name proctor \
  --gpus all \
  -p 8000:8000 \
  --restart unless-stopped \
  krushang08/proctor-webrtc:latest
```

### Step 7 — Verify backend is running
```bash
docker logs -f proctor
# Should see: "ObjectDetector ready device=cuda half=True"
# Should see: "Uvicorn running on http://0.0.0.0:8000"

curl http://localhost:8000/metrics
# Should return JSON with system metrics
```

### Step 8 — Update frontend .env.local
```
NEXT_PUBLIC_BACKEND_URL=http://<elastic-ip>:8000
```
Note: HTTP (not HTTPS) — EC2 direct access. Camera/mic on candidate page requires HTTPS
or localhost. See Section 7 for the HTTPS solution.

### Step 9 — Test single user
```
http://localhost:3000/candidate  →  start exam (WebRTC connects)
http://localhost:3000/admin      →  see session card appear
```

---

## 5. What Has Been Built & Completed

### Backend Features (all working)
- [x] FastAPI + aiortc WebRTC signaling (`POST /offer`)
- [x] YOLOv8 batch inference (GPU, FP16, warmup, auto device selection)
- [x] MediaPipe FaceMesh (head pose, gaze, lip movement, blink detection)
- [x] Silero-VAD audio monitoring via WebRTC push mode
- [x] Risk scoring engine (fixed + decaying buckets, NORMAL→WARNING→HIGH_RISK→TERMINATED)
- [x] Alert engine with cooldowns and SSE streaming
- [x] Debug overlay toggle with CV2 annotations
- [x] Tab switch detection + auto-terminate at 3 switches
- [x] 5-minute hard exam time limit (RISK_SESSION_DURATION_S = 300)
- [x] Metrics endpoint (`GET /metrics`) + System report (`GET /system/report`)
- [x] Proof capture (JPEG + WAV on alert)
- [x] Structured JSON logging with rotation
- [x] **Trickle ICE** — sends WebRTC offer immediately, candidates trickle via POST /ice-candidate/{pc_id}
- [x] **8-second grace period** on WebRTC disconnected state before marking session ended
- [x] **MediaPipe latency tracking** — `metrics.record_mediapipe_latency()` called after gather()
- [x] **Audio VAD latency tracking** — `metrics.record_audio_latency()` called after each silero inference
- [x] Both exposed in `GET /metrics` under `"mediapipe"` and `"audio"` keys

### Frontend Features (all working)
- [x] Admin dashboard (`/admin`) — live session cards
- [x] Session detail (`/admin/session/[pc_id]`) — snapshot, debug toggle, SSE alerts
- [x] System monitor (`/admin/monitor`) — live metrics + system report
- [x] Candidate page (`/candidate`) — WebRTC, tab switch detection, window blur, copy-paste disabled
- [x] Trickle ICE implementation — `onicecandidate` sends to `/ice-candidate/{pc_id}`
- [x] 8s disconnect grace — `onconnectionstatechange` with setTimeout

### Load Test Tools (`load_test/`)
- [x] `client.py` — single WebRTC client streaming a video file
- [x] `runner.py` — N concurrent clients with live metrics dashboard
- [x] `benchmark.py` — multi-level benchmark (1,5,10,25,50 clients) with HTML report
  - Charts: tick latency, YOLO latency, CPU, 10Hz maintenance, GPU (if available)
  - **NEW**: MediaPipe latency chart + Audio VAD latency chart
  - **NEW**: MP Avg(ms) and VAD Avg(ms) columns in results table
  - Bottleneck analysis: identifies which resource saturates first
- [x] `report.py` — summary report generator for runner.py output
- [x] `test_video.mp4` — test video exists (no audio track)

---

## 6. Recent Code Changes (This Session — 2026-03-19)

| File | Change |
|------|--------|
| `Proctor-webRTC/core/metrics.py` | Added `_mediapipe_latencies`, `_audio_latencies` deques; `record_mediapipe_latency()`, `record_audio_latency()` methods; exposed under `"mediapipe"` and `"audio"` in snapshot() |
| `Proctor-webRTC/core/proctor_coordinator.py` | Times `asyncio.gather(*mp_tasks)` wall clock; records via `metrics.record_mediapipe_latency()` only when real MediaPipe work ran |
| `Proctor-webRTC/core/audio_monitor.py` | Times each silero VAD `model()` call in both `_run()` and `_run_vad_only()`; records via `metrics.record_audio_latency()` |
| `load_test/benchmark.py` | Reads `mediapipe.lat_avg_ms` and `audio.lat_avg_ms` from /metrics; computes stats; prints MP/Audio lines in summary; adds MP+Audio chart and table columns to HTML report |
| `load_test/runner.py` | Fixed field names: `yolo.lat_avg_ms`, `system.cpu_percent`, `system.mem_rss_mb` |

---

## 7. HTTPS for Camera Access (Important)

Browsers block `getUserMedia()` (camera/mic) on plain HTTP except for `localhost`.
Since EC2 backend is accessed via `http://<ip>:8000`, the candidate page needs a workaround:

**Option A — SSH Tunnel (for testing)**
```bash
ssh -N -L 8000:localhost:8000 -i proctor-key.pem ubuntu@<elastic-ip>
# Then set: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
# localhost is trusted, getUserMedia works
```

**Option B — Self-signed cert + nginx (for production-like testing)**
```bash
# On EC2 instance
sudo apt install nginx certbot -y
# Generate self-signed cert
openssl req -x509 -nodes -days 365 -newkey rsa=2048 \
  -keyout /etc/ssl/proctor.key -out /etc/ssl/proctor.crt \
  -subj "/CN=<elastic-ip>"
# Configure nginx reverse proxy with SSL on 443 → localhost:8000
```
Then set `NEXT_PUBLIC_BACKEND_URL=https://<elastic-ip>` and accept the browser SSL warning once.

**Option C — Use ngrok (quickest for demo)**
```bash
# On EC2
ngrok http 8000
# Gives: https://xxxx.ngrok.io → use this as NEXT_PUBLIC_BACKEND_URL
```

---

## 8. Optimization Roadmap (Planned — Not Yet Implemented)

Discussed but not yet implemented. Do these after benchmark data confirms bottlenecks:

### Phase 1 — High impact, low effort (do regardless of benchmark)
1. **uvloop** — drop-in asyncio replacement, 2–4× I/O throughput
   ```bash
   pip install uvloop
   ```
   ```python
   # main.py — one line
   import uvloop
   asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
   ```

2. **Adaptive MediaPipe frame rate** — run at 5Hz instead of 10Hz when >10 sessions active
   - Duration gates are 1.5–2s; 5Hz gives 7–10 samples per window, sufficient
   - ~2× MediaPipe CPU capacity
   - In `proctor_session.py`: skip MediaPipe every other tick when load is high

### Phase 2 — After benchmark data
3. **TensorRT export** — if YOLO is the bottleneck (3–5× speedup)
   ```python
   model = YOLO("finalBestV5.pt")
   model.export(format="engine", device=0, half=True, imgsz=640)
   # Then load: YOLO("finalBestV5.engine")
   ```
   Note: .engine file is GPU-specific (T4). Rebuild if instance type changes.

4. **imgsz=320 at load time** — if YOLO throughput needs improvement
   ```python
   self.model.overrides['imgsz'] = 320  # set at load, never at inference time
   ```
   Test detection quality (small objects like earbuds) before committing.

5. **Pre-downscale frames for MediaPipe** — reduce 720p→480p before passing to FaceMesh
   (MediaPipe resizes to 192×192 internally anyway, no quality loss)

### Why NOT InsightFace for GPU MediaPipe
- InsightFace is a face recognition library — 5-point or 68-point landmarks only
- We need 468 landmarks for: lip MAR (40 lip points), EAR/blink (16 eye points), head pose PnP
- MediaPipe's FaceMesh model is 4MB — too small for GPU transfer overhead to be worth it
- MediaPipe on CPU with XNNPACK int8 is ~8ms/frame, already very fast
- More CPU cores (g4dn.2xlarge = 8 cores) is better value than GPU-izing MediaPipe

---

## 9. EC2 Instance Type Reference (For Future Scaling)

| Instance | vCPU | RAM | GPU | Est. max users* | On-Demand/hr |
|---|---|---|---|---|---|
| g4dn.xlarge | 4 | 16 GB | T4 16GB | 25–30 | $0.586 |
| g4dn.2xlarge | 8 | 32 GB | T4 16GB | 45–50 | ~$0.752 |
| g4dn.4xlarge | 16 | 64 GB | T4 16GB | 60–70 | ~$1.204 |
| g5.xlarge | 4 | 16 GB | A10G 24GB | 25–30 | ~$1.006 |
| g6.2xlarge | 8 | 32 GB | L4 24GB | 50–60 | ~$1.011 |

*Estimates assume TensorRT + adaptive MediaPipe optimizations applied.

**Recommendation**: Start with g4dn.xlarge, benchmark, then upgrade to g4dn.2xlarge if
MediaPipe CPU is the bottleneck (more cores, same T4 GPU, only 28% more expensive).
Avoid G5 unless benchmark proves YOLO is the bottleneck.

---

## 10. Critical Bug History (Do Not Re-Introduce)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `mediapipe AttributeError: no attribute 'solutions'` | pip installed mediapipe 0.11+ | Pin: `mediapipe==0.10.11` + `protobuf==3.20.3` |
| `ModuleNotFoundError: pyaudio` on server startup | Top-level `import pyaudio` | Made lazy import inside `_run()` only |
| `torchaudio CUDA version mismatch` | torchaudio from default PyPI (cu121) vs torch cu124 | Install all three together: `torch torchvision torchaudio --index-url .../cu124` |
| YOLO bounding boxes in wrong position | Pre-resizing frames + passing `imgsz` at runtime = double transform | Never pre-resize; never pass `imgsz` at inference time |
| Backend freeze on startup | `model.half()` + warmup in `__init__()` blocked asyncio | Warmup in `_schedule_warmup()` daemon thread |
| Tab switch not terminating at 3 | Browser suspended fetch when tab hidden | `fetch(..., keepalive: true)` |
| Alert flooding after termination | Tick loop kept processing | `if self.risk.terminated: return` in `session.update()` |
| ERR_EMPTY_RESPONSE on /offer | SSL certs present → HTTPS mode, frontend using HTTP | `mv ~/key.pem ~/key.pem.bak` to disable HTTPS mode |
| WebRTC state→failed on RunPod | RunPod blocks all UDP — no media path | Migrated to AWS EC2 with UDP 10000-60000 open |
| CORS 403 on RunPod proxy URL | FortiGuard IPS at office blocks proxy.runpod.net | Migrated to AWS EC2 direct IP access |

---

## 11. Trickle ICE Implementation (Already Done)

WebRTC now uses Trickle ICE — offer sent immediately, candidates trickle in. Reduces connection time from 3–5s to <1s.

**Backend** (`server.py`):
- `_pcs_by_id: dict[str, RTCPeerConnection]` — stores PCs by device_id for ICE routing
- `POST /ice-candidate/{pc_id}` endpoint — receives browser ICE candidates, calls `pc.addIceCandidate()`
- Uses `aiortc.sdp.candidate_from_sdp()` to parse candidate strings

**Frontend** (`frontend/app/candidate/page.tsx`):
- Sends offer immediately (no `waitGathering` / `iceGatheringState` wait)
- `pc.onicecandidate = ({ candidate }) => api.sendIceCandidate(answer.device_id, candidate.toJSON())`
- 8-second disconnect grace: `onconnectionstatechange` with `setTimeout(..., 8000)` on "disconnected"

**STUN only** (no TURN needed for EC2):
```python
_ICE_SERVERS = RTCConfiguration(iceServers=[
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
])
```
EC2 security group allows UDP — STUN (IP discovery) is sufficient. TURN (relay) was only needed when UDP was completely blocked (RunPod).

---

## 12. Load Testing Workflow (After EC2 is Running)

### Setup
```bash
cd load_test
pip install -r requirements.txt
```

### Test video with audio (current test_video.mp4 has no audio)
```bash
# Add silent audio track to existing video
ffmpeg -i test_video.mp4 -f lavfi -i anullsrc=r=16000:cl=mono \
  -c:v copy -c:a aac -shortest -y test_video_with_audio.mp4
```

### Run benchmark (main tool)
```bash
python benchmark.py \
  --url http://<elastic-ip>:8000 \
  --video test_video_with_audio.mp4 \
  --levels 1,5,10,25,50 \
  --warmup 30 \
  --steady 120
# Output: benchmark_<timestamp>.html + .json
```

Benchmark HTML report shows:
- Tick latency vs concurrency
- YOLO latency vs concurrency
- **MediaPipe latency vs concurrency** (new)
- **Audio VAD latency vs concurrency** (new)
- CPU/RAM/GPU usage vs concurrency
- 10Hz maintenance % vs concurrency
- Bottleneck analysis: which resource saturates first

### Run simple load test
```bash
python runner.py \
  --url http://<elastic-ip>:8000 \
  --video test_video_with_audio.mp4 \
  --clients 10 \
  --duration 300
```

---

## 13. GitHub Repository

- **Repo**: https://github.com/krushangshah18/ProctoDeployBackendGPU.git
- Push latest changes before switching machines:
```bash
git add Proctor-webRTC/core/metrics.py \
        Proctor-webRTC/core/proctor_coordinator.py \
        Proctor-webRTC/core/audio_monitor.py \
        load_test/benchmark.py \
        load_test/runner.py \
        SESSION_CONTEXT.md
git commit -m "Add MediaPipe+audio latency tracking; update session context for EC2 migration"
git push origin main
```

---

## 14. Immediate Next Steps (In Order)

```
1. [ ] AWS quota approved (email notification) — "Running On-Demand G and VT instances" → 4 vCPUs
2. [ ] Launch EC2 instance (proctor-backend, g4dn.xlarge, Mumbai)
3. [ ] Allocate + Associate Elastic IP
4. [ ] SSH in: ssh -i proctor-key.pem ubuntu@<elastic-ip>
5. [ ] Verify: nvidia-smi (should show T4)
6. [ ] Verify: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
7. [ ] Run container: docker pull krushang08/proctor-webrtc:latest && docker run -d --gpus all -p 8000:8000 --name proctor krushang08/proctor-webrtc:latest
8. [ ] Verify backend: docker logs -f proctor (should see cuda device ready)
9. [ ] Update frontend/.env.local: NEXT_PUBLIC_BACKEND_URL=http://<elastic-ip>:8000
10.[ ] Set up HTTPS (SSH tunnel OR self-signed cert) for getUserMedia on candidate page
11.[ ] Test single user end-to-end (candidate → admin → session card appears)
12.[ ] Add silent audio to test video (ffmpeg command above)
13.[ ] Run benchmark: python benchmark.py --url http://<elastic-ip>:8000 --video test_video_with_audio.mp4 --levels 1,5,10 --warmup 30 --steady 120
14.[ ] Analyse HTML report — identify bottleneck (MediaPipe CPU vs YOLO GPU)
15.[ ] Implement Phase 1 optimizations (uvloop + adaptive MediaPipe)
16.[ ] Re-benchmark to confirm gains
17.[ ] Decide on g4dn.xlarge vs g4dn.2xlarge based on real data
```

---

## 15. Quick Reference Commands

```bash
# SSH into EC2
ssh -i /path/to/proctor-key.pem ubuntu@<elastic-ip>

# Check GPU
nvidia-smi

# Docker container management
docker ps                          # check running
docker logs -f proctor             # follow logs
docker restart proctor             # restart container
docker exec -it proctor bash       # shell inside container

# Health checks
curl http://<elastic-ip>:8000/metrics
curl http://<elastic-ip>:8000/sessions
curl http://<elastic-ip>:8000/system/report

# SSH tunnel for HTTPS workaround
ssh -N -L 8000:localhost:8000 -i proctor-key.pem ubuntu@<elastic-ip>
# Then: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Frontend
cd frontend && npm run dev         # http://localhost:3000

# Push code changes to EC2 without Docker rebuild (no new dependencies)
scp -i proctor-key.pem Proctor-webRTC/server.py ubuntu@<elastic-ip>:/home/ubuntu/server.py
ssh -i proctor-key.pem ubuntu@<elastic-ip>
  docker cp server.py proctor:/app/server.py
  docker restart proctor

# Full Docker rebuild + push (when requirements.txt or Dockerfile changes)
cd Proctor-webRTC
docker build -t krushang08/proctor-webrtc:latest .
docker push krushang08/proctor-webrtc:latest
# Then on EC2: docker pull + docker restart proctor
```

---

*Last updated: 2026-03-19. RunPod fully abandoned. AWS EC2 g4dn.xlarge configured and ready to launch. Blocked on vCPU quota increase request — waiting for AWS approval.*
