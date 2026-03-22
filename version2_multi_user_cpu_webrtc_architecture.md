# Version 2 Architecture

Source implementation: `D:\VS Code\multiCPU\NextAndFastApiWithWebRTC`

This document explains the second implementation in depth. Version 2 evolves the original single-user local proof of concept into a browser-based, multi-user, CPU-oriented proctoring platform built around WebRTC, FastAPI, and Next.js. The core cheating-detection ideas remain similar, but the architecture changes significantly to support multiple concurrent sessions, browser clients, admin monitoring, and centralized reporting.

## 1. Goal of this version

Version 2 is the first serious multi-session system. It answers the question: "How do we take the version 1 proctoring logic and run it for several remote candidates through the browser, without requiring a local native camera loop on the server side?"

It optimizes for:

- browser-native candidate onboarding
- centralized backend control
- concurrent session handling on CPU
- shared object detection across sessions
- admin visibility and report access

It does not yet optimize for:

- high user counts
- GPU-heavy inference
- hardened cloud deployment
- live runtime retuning of all policy knobs

## 2. Major architectural shift from version 1

Version 1 was a local machine loop. Version 2 becomes a distributed client-server system.

Main changes:

- the candidate now runs in the browser instead of a local OpenCV app
- video and audio arrive through WebRTC
- the backend owns proctoring state for many users
- object detection is shared and batched across sessions
- the frontend is split into candidate, admin, monitor, and report views
- proof and reporting are stored centrally

This is the point where the implementation stops being just a detector experiment and becomes an actual product architecture.

## 3. Repository structure and system layout

The project has two major parts:

- backend: `D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC`
- frontend: `D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend`

Backend responsibilities:

- receive browser media via WebRTC
- run detection pipelines
- track session state
- compute risk and alerts
- expose APIs, SSE streams, snapshots, reports, and metrics

Frontend responsibilities:

- acquire camera and microphone permissions
- establish peer connections to the backend
- report browser tab/visibility violations
- display local warnings and session status
- give admins live oversight and report access

## 4. Backend runtime architecture

The backend entry flow is:

- [`main.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\main.py) parses startup flags such as `--device`, `--half`, `--warmup`, and `--port`.
- [`server.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\server.py) starts FastAPI.
- the FastAPI lifespan hook builds configuration and starts a shared [`ProctorCoordinator`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\proctor_coordinator.py).

Important config defaults from [`config.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\config.py):

- tick rate: `10 Hz`
- max sessions: `3`
- YOLO device: `cpu`
- half precision: `False`
- YOLO warmup frames: `0`
- minimum VRAM guard for GPU mode: `1.5 GB`

The max session value is conservative because this build is CPU oriented. The architecture is multi-user, but the deployment target is still modest hardware.

## 5. Session creation and WebRTC flow

The central browser-to-backend handshake happens through `/offer`.

Session setup flow:

1. Candidate page acquires camera and microphone with browser media APIs.
2. The frontend creates an `RTCPeerConnection`.
3. Local audio and video tracks are added to the connection.
4. The frontend waits for ICE gathering to complete.
5. The SDP offer is posted to `/offer`.
6. The backend creates a new `RTCPeerConnection`, a new `ProctorSession`, and attaches media-track handlers.
7. The backend returns the SDP answer plus identifiers such as `pc_id` and `report_id`.
8. The frontend sets the remote description and opens an SSE stream for live session events.

Why WebRTC is used:

- it is the browser-native way to stream real-time audio and video
- it avoids forcing candidates to install software
- it provides low-latency media delivery

## 6. Media ingestion on the backend

### 6.1 Video path

`server.py` defines `VideoAnalyzerTrack`, which wraps the inbound video track.

Its responsibilities:

- receive frames from WebRTC
- convert them into a usable image format
- store the latest frame for session processing
- compute basic stats such as FPS, jitter, and resolution
- create JPEG snapshots at roughly `2 Hz` for admin views

This design decouples transport from detection. The track receiver only ensures the latest media is available; it does not perform heavy detection itself.

### 6.2 Audio path

`server.py` also defines `AudioAnalyzerTrack`.

Its responsibilities:

- receive WebRTC audio frames
- resample to mono `16 kHz`
- convert to PCM chunks
- push the data into the session audio monitor ring buffer

This is a key change from version 1. In version 1, the server itself captured the local machine microphone. In version 2, the browser media stream becomes the source of truth.

## 7. Central coordinator and shared inference

[`core/proctor_coordinator.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\proctor_coordinator.py) is the most important new architectural piece.

Its role is to coordinate all active sessions from one shared control loop.

Per tick, the coordinator:

1. snapshots the active sessions
2. gathers the latest frames from sessions that are ready
3. runs batch YOLO on all collected frames
4. runs MediaPipe processing for sessions that need it
5. calls `session.update(...)` for each session with the new results

Why this coordinator exists:

- without it, each session would run its own independent detector loop
- that would duplicate model loading, waste CPU, and make batch optimization impossible
- central coordination gives one place to enforce tick cadence and shared resource use

This is the bridge from a single-user prototype to a resource-managed multi-user service.

## 8. Per-session architecture

Each candidate session is represented by [`ProctorSession`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\proctor_session.py).

Each session owns:

- identity and lifecycle state
- current risk totals and event history
- SSE subscribers for live updates
- `FaceMeshProvider`
- `HeadPoseDetector`
- `LipDetector`
- `ObjectTemporalTracker`
- `HeadTracker`
- `LivenessDetector`
- `AudioMonitor`
- `SpeakerAudioDetector`
- `RiskEngine`
- `ProofWriter`

This separation is important:

- heavyweight model resources such as YOLO are shared at coordinator level
- personal behavioral state such as object history, fake-presence history, and risk totals stay session local

That split gives scalability without losing behavioral continuity.

## 9. MediaPipe reuse and CPU optimization

Version 2 introduces [`detectors/face_mesh_provider.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\detectors\face_mesh_provider.py).

Why this matters:

- in version 1, multiple components could end up re-running landmark extraction logic
- in version 2, FaceMesh output is shared within the session
- `HeadPoseDetector` and `LipDetector` can both use the same landmark set

This is an important CPU optimization because FaceMesh is a non-trivial cost. Reusing landmarks prevents repeated work inside the same tick.

`ProctorSession.needs_mediapipe()` adds another optimization:

- if the current detection configuration disables all face-dependent logic
- or if the current state does not require FaceMesh work
- the session can skip MediaPipe for that tick

That matters on CPU because every skipped face inference directly increases capacity.

## 10. Detailed per-session processing flow

When `ProctorSession.update(...)` runs, the session processes the candidate state in a layered sequence.

### 10.1 Frame guards and timing

The session first handles base timing and frame validity:

- tracks effective tick FPS
- detects dark or invalid frames
- updates recent session timestamps

These guards prevent later components from treating bad transport states as meaningful behavior.

### 10.2 Object interpretation

YOLO detections are normalized into session-level flags:

- person count
- phone presence
- book presence
- earbud or headphone presence

Then the `ObjectTemporalTracker` smooths these detections over time.

Important version 2 change in [`core/object_tracker.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\object_tracker.py):

- the tracker is now time based instead of frame-count based
- default window: `1.5s`
- minimum votes are derived from measured FPS and configured ratios

Why this is better:

- in multi-user systems, FPS differs by candidate and by network conditions
- a 15-frame window means different real durations at 5 FPS and 20 FPS
- a 1.5-second window preserves policy meaning regardless of transport variability

### 10.3 Face and head analysis

The session uses shared landmarks to compute:

- yaw
- pitch
- gaze direction
- blink state
- face visibility
- partial-face state

The numeric thresholds remain broadly aligned with version 1:

- looking away threshold `2.0s`
- gaze threshold `1.5s`
- yaw `0.20`
- look down `0.13`
- look up `-0.10`
- gaze left `-0.13`
- gaze right `0.13`
- EAR threshold `0.20`

### 10.4 Liveness

`LivenessDetector` still evaluates motion variance and blink recency to detect fake presence. The logic is mostly carried forward from version 1 because the detection problem has not changed, only the transport has.

### 10.5 Lip and audio fusion

The `AudioMonitor` now runs in WebRTC mode through `start_webrtc_mode()`. Instead of reading a local microphone device, it analyzes audio chunks pushed by the audio track receiver.

The `SpeakerAudioDetector` then compares:

- VAD speech state
- lip speaking state
- face availability

This produces the `speaker_audio` event when room speech likely does not belong to the candidate.

### 10.6 Event normalization

The session derives normalized events such as:

- `phone`
- `book`
- `headphone`
- `multiple_people`
- `no_person`
- `fake_presence`
- `face_hidden`
- `partial_face`
- `looking_left`
- `looking_right`
- `looking_down`
- `speaker_audio`

These are then passed into the risk engine.

## 11. Scoring, warning, and alerting

[`core/risk_engine.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\risk_engine.py) keeps the same policy philosophy as version 1:

- fixed risk for serious evidence
- decaying risk for lower-certainty or recoverable suspicion
- duration gating
- cooldowns
- combination bonuses
- termination conditions

The same major thresholds are retained:

- warning threshold `30`
- high risk `60`
- admin review `100`

The version 2 backend still uses rules such as:

- phone repeated presence escalates fixed risk
- multiple people can terminate after long continuous presence
- no person can terminate after long continuous absence
- fake presence escalates in time-based stages
- gaze and posture cues add decaying risk
- suspicious cue combinations add extra context risk

This continuity is deliberate. Version 2 changes scale and transport, not the core proctoring policy.

### 11.1 Tab-switch reporting

A major new signal in version 2 is browser tab or focus loss.

The candidate frontend reports tab violations through `/tab_switch` using:

- `document.visibilitychange`
- `window.blur`
- `window.focus`

`ProctorSession.report_tab_switch()` forwards the event to the risk engine through `handle_tab_switch()`.

Observed scoring constants in [`settings/scoring.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\settings\scoring.py):

- tab-switch score: `15`
- terminate after `3` tab-switch occurrences

Important implementation note:

- code comments and policy intent indicate a warn-then-score-then-terminate progression
- the currently observed implementation in `handle_tab_switch()` adds score on non-terminal occurrences and terminates at the configured count

For documentation and audits, it is best to treat the current code as the source of truth and call out the intended policy separately.

## 12. Alert and SSE event procedure

Version 2 turns local alerts into server-pushed events.

The session publishes SSE messages such as:

- `warning`
- `alert`
- `session_end`

Why SSE is used:

- the frontend needs a simple one-way live event channel
- the server mostly pushes state outward rather than receiving continuous frontend commands
- SSE is simpler to manage than a second full-duplex channel for this use case

Candidate impact:

- the candidate page can show immediate warning banners
- the candidate can see risk escalation without polling

Admin impact:

- admin views can update timelines and risk status in near real time

## 13. Proof generation and reporting

[`utils/proof_writer.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\utils\proof_writer.py) is intentionally simpler than version 1.

Version 2 proof strategy:

- save JPEG proof for important visual events
- save WAV proof for speaker-audio events
- avoid heavy rolling video capture

Why proof is simplified:

- multi-user CPU architecture cannot afford the same proof overhead as the single-user prototype
- JPEG plus audio proof preserves enough evidence for review while being far cheaper than clip generation

This is a classic production tradeoff:

- less rich than the POC
- far more scalable

Reports are then exposed through backend APIs and rendered in the Next.js report pages.

## 14. Frontend architecture

### 14.1 Shared API layer

[`frontend/lib/api.ts`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\lib\api.ts) centralizes all backend requests:

- session creation
- tab-switch reporting
- session listing
- snapshots
- metrics
- reports
- exam configuration updates

This keeps the pages thin and consistent.

### 14.2 Candidate page

[`frontend/app/candidate/page.tsx`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\app\candidate\page.tsx) is the candidate runtime UI.

Its responsibilities:

- request camera and microphone permission
- build the peer connection
- send the WebRTC offer
- subscribe to SSE events
- report tab switches and focus loss
- block copy, cut, paste, and common clipboard shortcuts
- display warnings and connection state

Why clipboard blocking is included:

- it does not replace proctoring
- but it reduces obvious copy-paste cheating paths and complements browser-focus monitoring

### 14.3 Admin dashboard

[`frontend/app/admin/page.tsx`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\app\admin\page.tsx) polls `/sessions` roughly every `3s`.

It provides:

- active session cards
- current risk values
- ordering by severity
- toggles for exam detection config

This is the operational control room for the CPU deployment.

### 14.4 Session monitor

[`frontend/app/admin/session/[pc_id]/page.tsx`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\app\admin\session\[pc_id]\page.tsx) gives per-candidate inspection.

It provides:

- repeated snapshot polling, around `7 FPS`
- SSE event updates
- proof display
- debug overlays when enabled
- warnings and alerts timeline

### 14.5 Reports

The reports area uses:

- [`frontend/app/reports/page.tsx`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\app\reports\page.tsx)
- [`frontend/app/report/[report_id]/page.tsx`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\frontend\app\report\[report_id]\page.tsx)

These pages expose:

- completed report listing
- risk summaries
- warnings
- alerts
- attached proof

## 15. Metrics and observability

[`core/metrics.py`](D:\VS Code\multiCPU\NextAndFastApiWithWebRTC\Proctor-webRTC\core\metrics.py) records operational statistics such as:

- session counts
- alert and warning totals
- YOLO latency
- tick latency
- request latency
- system resource usage

Why metrics become essential in version 2:

- performance issues are no longer purely local
- one slow session can affect shared scheduling
- admin and deployment tuning require evidence

Metrics are the first step toward capacity planning.

## 16. Important design decisions and why they matter

### 16.1 Shared YOLO instead of per-session YOLO

This is the key CPU scaling move. Loading and running one detector per user would collapse quickly on commodity hardware.

### 16.2 Time-based smoothing

A multi-user network system cannot assume stable FPS. Converting temporal rules from frame windows to time windows preserves policy consistency.

### 16.3 Browser event reporting

Some cheating signals are easiest to capture in the browser, not the video stream. Tab switching is the clearest example.

### 16.4 Lightweight proof

Evidence is still necessary, but proof generation must fit multi-user CPU constraints.

### 16.5 Session-local behavioral state

Sharing detection infrastructure does not mean sharing candidate behavior history. Each session still needs independent temporal memory for fair scoring.

## 17. Limitations of version 2

Version 2 is a real multi-user system, but it is still constrained.

Main limitations:

- CPU-only operation limits concurrent users
- current config allows only a small number of sessions
- WebRTC and inference scheduling are not yet hardened for cloud/network edge cases
- proof is lighter than the original POC
- admin settings are more limited than in the later GPU version
- single backend instance monitoring is simpler than horizontally distributed monitoring

These issues are what version 3 addresses.

## 18. What version 2 contributes to the overall evolution

Version 2 is the architectural turning point of the project.

It proves that the proctoring logic can be:

- detached from local-device assumptions
- attached to browser media streams
- run concurrently for multiple candidates
- monitored by administrators in real time
- reported centrally

If version 1 is the policy blueprint, version 2 is the service blueprint. It introduces the control plane, transport model, and session model that later GPU and AWS deployment work builds upon.
