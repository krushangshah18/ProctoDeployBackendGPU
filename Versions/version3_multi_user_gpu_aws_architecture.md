# Version 3 Architecture

Source implementation: `D:\VS Code\PocOnAWS\ProctoDeployBackendGPU`

This document explains the third implementation in depth. Version 3 is the GPU-oriented, AWS-ready evolution of the proctoring system. It keeps the session model and most policy ideas from version 2, but pushes the platform toward real deployment conditions: CUDA-backed inference, Docker packaging, WebRTC hardening, runtime tuning, multi-instance awareness, and load testing.

## 1. Goal of this version

Version 3 answers the question: "How do we take the multi-user WebRTC architecture and make it operationally viable on cloud GPU infrastructure?"

It optimizes for:

- GPU-backed multi-user inference
- stable browser-to-cloud WebRTC connectivity
- deployability on AWS EC2
- runtime tuning without code changes
- improved observability and benchmark tooling
- safer operations under real network conditions

This version is not just faster. It is more operationally mature.

## 2. Position in the evolution

Version 1 established:

- detection ideas
- scoring policy
- warnings, alerts, and proof logic

Version 2 established:

- WebRTC transport
- backend session model
- shared inference coordination
- candidate and admin frontend flows

Version 3 builds on both and adds:

- CUDA-first inference
- Dockerized deployment
- WebRTC protocol hardening for cloud environments
- runtime admin settings
- multi-backend frontend support
- load-test and benchmarking tooling

## 3. Deployment target and operating assumptions

The project notes in [`SESSION_CONTEXT.md`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\SESSION_CONTEXT.md) describe the intended environment:

- AWS EC2
- current target class around `g4dn.xlarge`
- NVIDIA T4 GPU
- Dockerized backend
- host networking for reliable WebRTC behavior
- STUN-assisted ICE negotiation

The same context document notes that a safe operating limit is around `5` users on the present target configuration. That number is important because it shows the project is capacity-aware rather than assuming theoretical GPU throughput is always usable in practice.

## 4. Backend runtime configuration

Important defaults from [`Proctor-webRTC/config.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\config.py):

- tick rate: `10 Hz`
- max sessions: `5`
- YOLO device: `cuda`
- half precision: `True`
- warmup frames: `3`
- minimum VRAM guard: `2.0 GB`
- inference image size: `640`
- MediaPipe stride: `3`
- risk session duration: `300s`
- dedicated `HEADPHONE_MIN_VOTES` control exists in this version

Compared with version 2, the config reflects a much more intentionally tuned runtime. The system is no longer simply "CPU but maybe GPU if available." It is explicitly GPU-oriented.

## 5. Containerization and startup model

[`Proctor-webRTC/Dockerfile`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\Dockerfile) packages the backend around:

- `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Python `3.11`
- CUDA-compatible `torch`, `torchvision`, and `torchaudio`
- copied weights and backend source

The container command is:

- `python main.py --half --warmup 3`

Why this matters:

- the deployment becomes reproducible
- CUDA and PyTorch compatibility are pinned intentionally
- GPU optimizations become part of the default startup contract rather than ad hoc flags

## 6. Entry point and startup behavior

[`main.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\main.py) is slimmer and more deployment focused than earlier versions.

Observed behavior:

- optional `uvloop` installation for faster event loop performance
- startup arguments for `--half`, `--warmup`, and `--port`
- GPU-oriented expectation rather than device auto-selection

This reflects an operational assumption: the container is supposed to run on hardware prepared for this workload, not on arbitrary developer laptops.

## 7. WebRTC hardening in the backend

[`server.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\server.py) contains some of the most important maturity improvements in version 3.

### 7.1 ICE and cloud connectivity

The backend introduces:

- configured `_ICE_SERVERS`
- STUN-aware peer connection setup
- an `_pcs_by_id` map
- `/ice-candidate/{pc_id}` for trickle ICE

Why this matters:

- cloud deployments do not behave like localhost
- candidates may be behind NATs, varying networks, and different browser timing conditions
- trickle ICE improves connection establishment robustness by exchanging candidates incrementally

### 7.2 SDP sanitation and codec handling

Version 3 strips `video/rtx` payloads from incoming SDP using `_strip_rtx_from_sdp()`.

It also forces VP8 codec preference after remote description.

Why:

- aiortc decoder issues and RTX negotiation edge cases caused real failures
- production systems often need defensive handling for codec combinations that are technically valid but unstable in practice

### 7.3 Safer receive loops

The track receive loops avoid `asyncio.wait_for(...)` and instead use `asyncio.wait({recv_task}, timeout=...)`.

Why this is important:

- cancelling `recv()` too aggressively can corrupt aiortc jitter-buffer behavior
- a timeout wrapper that preserves task integrity is safer than hard cancellation

This is a very practical example of deployment learning being translated into code architecture.

### 7.4 Recovery on stalls

The backend sends PLI when video stalls are detected.

Why:

- in real WebRTC sessions, video can freeze without fully disconnecting
- PLI helps request a fresh keyframe and recover the stream

### 7.5 CORS and error handling

The backend:

- supports CORS origins from environment configuration instead of unconditional `"*"`
- uses a global exception handler that also preserves required CORS headers

This is necessary for deployed frontends and safer cross-origin behavior.

## 8. Shared coordinator and GPU scheduling

[`core/proctor_coordinator.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\core\proctor_coordinator.py) extends the version 2 coordinator into a GPU-conscious scheduler.

Key responsibilities:

- own the shared CUDA YOLO detector
- process all active sessions on the central tick
- manage runtime settings in addition to exam detection toggles
- overlap work using `asyncio.gather`
- support staggered MediaPipe execution

### 8.1 Shared CUDA detector

The coordinator initializes one shared detector for batch inference. This keeps GPU memory use controlled and maximizes the value of batching.

### 8.2 Runtime settings model

Unlike version 2, the coordinator stores a richer `runtime_settings` object. This allows admins to tune system behavior without modifying code and redeploying.

### 8.3 Overlapping work

The tick loop overlaps parts of the processing pipeline with `asyncio.gather`.

Why:

- GPU work, CPU landmark work, and per-session update logic do not always need to be serialized
- careful overlap improves throughput and reduces dead time

## 9. Staggered MediaPipe strategy

One of the most important scaling ideas in version 3 is `MEDIAPIPE_STRIDE`.

Observed default:

- `MEDIAPIPE_STRIDE = 3`

What this means:

- not every session gets fresh MediaPipe processing on every tick
- sessions are partitioned by tick index
- only the selected subset gets new landmarks on a given tick
- other sessions reuse their last valid MediaPipe result

Why this exists:

- YOLO is on GPU and scales better with batching
- MediaPipe remains CPU-side and becomes a bottleneck earlier
- staggering reduces CPU pressure while keeping face-derived signals sufficiently fresh for a `10 Hz` system

This is a deliberate accuracy-throughput tradeoff:

- some face signals are slightly less fresh
- overall multi-user capacity improves materially

## 10. Per-session architecture in version 3

[`core/proctor_session.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\core\proctor_session.py) remains structurally close to version 2, but adds important deployment-oriented refinements.

Each session still owns:

- local behavioral history
- risk state
- event subscribers
- FaceMesh consumer objects
- audio and liveness logic
- proof writer

Important refinements:

- `HeadPoseDetector` is constructed with thresholds injected from config
- headphone vote tuning is configurable separately
- `_tick_fps` is initialized to `10.0` and updated with a faster EMA
- comments explain fixes for late-joining users missing detections

Why explicit threshold injection is valuable:

- settings become easier to expose through admin APIs
- the detector becomes less dependent on hidden module constants
- runtime tuning becomes more transparent and maintainable

## 11. Detection pipeline in version 3

The logical proctoring sequence is still:

1. receive browser media
2. store latest frame and audio chunks
3. batch YOLO across active sessions
4. run fresh or reused MediaPipe analysis
5. compute head, gaze, mouth, liveness, and object state
6. smooth detections over time
7. generate normalized suspicious events
8. score those events
9. emit warnings, alerts, or session termination
10. persist proof and report data

The important shift is not the policy sequence, but the way compute is scheduled under cloud constraints.

## 12. GPU object detection details

[`detectors/object_detector.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\detectors\object_detector.py) is explicitly GPU-first.

Observed behavior:

- `_ensure_cuda()` raises if CUDA is not available
- model image size override is set through `model.overrides["imgsz"]`
- FP16 inference is enabled by default
- batch inference remains the core usage pattern

Why each choice matters:

- hard CUDA enforcement prevents silent fallback to an underpowered CPU path
- image-size control is essential for balancing accuracy and latency
- FP16 reduces GPU bandwidth and memory pressure, often improving throughput
- batching is the main reason shared GPU inference is worthwhile

## 13. Risk engine, scoring, warnings, and alerts

The policy model remains broadly consistent with versions 1 and 2:

- fixed risk and decaying risk still coexist
- duration gates are still important
- repeated phone, no-person, fake-presence, multiple-people, gaze, and speaker-audio patterns still drive escalation
- termination still exists for certain severe continuous conditions

What changes in version 3 is less the policy itself and more the ability to tune policy values live through admin settings.

This is important operationally because once the system is deployed:

- exam contexts differ
- hardware behavior differs
- network quality differs
- false-positive tolerance differs

A runtime-tunable policy engine is far more practical than a compile-time one.

## 14. Admin settings API and live tuning

Version 3 introduces `/admin/settings` GET and POST endpoints in [`server.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\server.py).

These endpoints expose a richer set of controls, including:

- detection toggles
- object thresholds
- head and gaze thresholds
- smoothing and voting values
- scoring parameters
- cooldowns
- runtime behavior controls

Why this is a major architectural step:

- operators no longer need code edits for common tuning tasks
- deployment experimentation becomes far faster
- one backend can be tuned differently from another backend instance

This turns the proctoring engine into a configurable service rather than a fixed binary.

## 15. Frontend evolution in version 3

### 15.1 Multi-backend API support

[`frontend/lib/api.ts`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\frontend\lib\api.ts) introduces `createApi(base)` and `BACKEND_URLS`.

Why:

- the frontend can talk to more than one backend instance
- admin views can aggregate or switch between instances
- this prepares the system for horizontal deployment patterns

### 15.2 Candidate page improvements

[`frontend/app/candidate/page.tsx`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\frontend\app\candidate\page.tsx) adds:

- trickle ICE candidate posting
- STUN-aware peer setup
- an `8s` disconnected grace window

Why the grace window exists:

- cloud sessions may see transient network drops
- brief connection instability should not immediately destroy the exam

### 15.3 Admin dashboard improvements

[`frontend/app/admin/page.tsx`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\frontend\app\admin\page.tsx) can merge sessions from multiple backend URLs and keep per-backend context.

This is the first clear move toward multi-instance operations.

### 15.4 Monitor page

[`frontend/app/admin/monitor/page.tsx`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\frontend\app\admin\monitor\page.tsx) supports:

- backend switching
- per-instance metrics
- system report inspection

### 15.5 Admin settings UI

[`frontend/app/admin/settings/page.tsx`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\frontend\app\admin\settings\page.tsx) exposes a full tuning interface for the backend settings APIs.

This closes the loop between runtime configurability and operational usability.

## 16. Metrics and observability

[`core/metrics.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\Proctor-webRTC\core\metrics.py) extends the version 2 metrics model.

Version 3 adds or emphasizes:

- MediaPipe latency
- audio latency tracking
- richer snapshot metrics for system monitoring

This is necessary because GPU deployments fail in more complex ways:

- GPU may be fast while CPU-side MediaPipe is saturated
- audio processing may drift under load
- network and inference delays may interact

Without richer metrics, it is hard to know which subsystem is really limiting throughput.

## 17. Load testing and benchmarking

Version 3 includes a dedicated load-test stack:

- [`load_test/client.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\load_test\client.py)
- [`load_test/runner.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\load_test\runner.py)
- [`load_test/benchmark.py`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\load_test\benchmark.py)

What these tools do:

- simulate candidate clients using aiortc
- stream prerecorded media
- create concurrent load against one or two backends
- poll metrics during the run
- generate JSON and HTML benchmark outputs

Why this matters:

- capacity planning cannot rely on guesswork
- cloud tuning requires repeatable experiments
- comparing two backend versions becomes possible with evidence instead of anecdotes

This is a major sign of engineering maturity.

## 18. Known operational learnings captured by the codebase

[`SESSION_CONTEXT.md`](D:\VS Code\PocOnAWS\ProctoDeployBackendGPU\SESSION_CONTEXT.md) documents important issues that influenced the architecture:

- MediaPipe version pinning problems
- lazy `pyaudio` import fixes
- Torch and torchaudio compatibility mismatches
- bounding-box coordinate errors caused by image-size handling
- startup freezes related to half precision and warmup placement
- tab-switch keepalive issues
- alert flooding after termination
- SSL confusion in deployment
- Docker bridge networking causing WebRTC connection problems
- frame-freeze issues caused by `wait_for`
- RTX decoder crashes
- `_tick_fps` initialization bugs
- ProcessPool experiments being reverted

This matters because version 3 is not only shaped by theory. It is shaped by actual failure modes encountered during deployment work.

## 19. Why this architecture was chosen

### 19.1 GPU where it matters, CPU where it is acceptable

Object detection benefits heavily from GPU batching, so the system moves YOLO there. MediaPipe remains CPU-bound, and the system works around that with stride-based scheduling instead of pretending the bottleneck does not exist.

### 19.2 Runtime configurability over static tuning

Cloud systems need rapid adjustment. Rebuilding the image for every threshold change would slow operations too much.

### 19.3 Protocol hardening over idealized WebRTC assumptions

Production WebRTC is messy. The backend includes very practical fixes because reliability matters more than elegance.

### 19.4 Capacity validated by benchmark tooling

A service that handles proctoring risk must be predictable under load. Dedicated benchmark tooling is the right answer.

## 20. Remaining constraints in version 3

Version 3 is the most mature of the three, but it still has natural constraints.

Main constraints:

- MediaPipe remains a scaling bottleneck even with stride scheduling
- the documented safe user count is still modest on the chosen GPU instance
- multi-instance support exists in the frontend, but full distributed orchestration is still a separate problem
- real deployment quality still depends on network conditions, ICE reliability, and browser behavior

In other words, version 3 is deployment ready for controlled scale, not unlimited scale.

## 21. What version 3 contributes to the overall system

Version 3 is the operational blueprint for the project.

It proves that the system can be:

- containerized
- GPU accelerated
- deployed on AWS
- tuned live by admins
- observed through metrics
- validated under synthetic load
- hardened against real WebRTC edge cases

If version 1 is the detection and policy blueprint, and version 2 is the service blueprint, then version 3 is the production deployment blueprint.
