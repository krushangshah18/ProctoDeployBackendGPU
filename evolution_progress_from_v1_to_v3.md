# Evolution Progress Across Versions

This document explains the project as an engineering progression rather than a strict feature-by-feature comparison. The focus is on how each version exposed practical limits, why those limits mattered, and what architectural changes were introduced in the next version to remove or reduce them.

The key idea is simple:

- version 1 proved the full proctoring logic
- version 2 re-architected that logic so multiple users could be handled centrally
- version 3 hardened and accelerated the multi-user system for GPU and cloud deployment

## 1. Version 1: proving the full proctoring pipeline

The first version was built as a single-user, CPU-oriented proof of concept. Its job was not to serve many candidates at once. Its job was to answer a more basic question:

"Can one machine observe one candidate through webcam and microphone, detect suspicious behavior, score it intelligently, produce warnings and alerts, and save evidence?"

That is why version 1 was built around a very direct local runtime:

- local webcam capture through OpenCV
- local microphone capture through a background audio thread
- one synchronous frame-processing loop
- one candidate state in memory
- one local UI window with overlays
- one local proof-writing path

This version was extremely important because it established the core logic of the entire system:

- object detection
- face and head-pose analysis
- gaze and looking-away logic
- lip-motion analysis
- speaker-audio inference
- liveness detection
- temporal smoothing
- risk scoring
- warnings, alerts, and termination conditions
- proof collection

In other words, version 1 gave us the proctoring brain.

## 2. What made version 1 single-user specific

Version 1 was not "single-user" only because of hardware size. It was single-user because of how the architecture itself was shaped.

### 2.1 Local-device assumptions

The system assumed the server process and the candidate environment were effectively the same machine.

Examples:

- camera input came from `cv2.VideoCapture(0)`
- audio came from a locally attached microphone device
- output was rendered with `cv2.imshow`

That means the pipeline was built around direct access to one machine's peripherals, not around receiving remote media streams from many browsers.

### 2.2 One tightly coupled processing loop

The main logic lived in a single loop that processed one frame at a time for one person. This was excellent for debugging, but it meant:

- no session scheduler existed
- no concurrent candidate state management existed
- no shared batching model existed

The loop was essentially:

1. read one frame
2. detect objects
3. run face analysis
4. run lip analysis
5. consult audio state
6. update one risk engine
7. draw overlays
8. show the result locally

That flow is naturally single-user because the entire process assumes exactly one current subject.

### 2.3 No transport layer

Version 1 had no browser-to-backend media transport design. There was:

- no WebRTC
- no peer connection lifecycle
- no ICE flow
- no session signaling
- no browser event reporting

Without a transport layer, the system could not serve remote candidates in a controlled centralized way.

### 2.4 No concept of a session object per candidate

Version 1 had state, but that state was global to the running application. It was not modeled as:

- session A
- session B
- session C

There was no need yet for:

- per-user timestamps
- per-user event logs
- per-user SSE streams
- per-user proof and reports
- per-user cleanup and lifecycle handling

This is a major reason version 1 could not simply be "scaled up" by small edits. The architecture itself did not think in terms of independent candidates.

### 2.5 No centralized admin control

Version 1 was focused on local observation and debugging. It did not have:

- an admin dashboard
- a candidate page
- a report page
- live session listing
- centralized risk monitoring

That was fine for a proof of concept, but it meant the system was not yet shaped like a deployable proctoring platform.

### 2.6 CPU usage pattern was acceptable only because there was one user

In version 1, it was acceptable to spend CPU more freely because only one candidate was being processed.

Examples:

- landmark-related work could be relatively expensive
- proof generation could be richer
- frame-based smoothing was acceptable
- local display overlays were useful

Those choices were helpful for correctness and debugging, but they do not scale naturally to many simultaneous users.

## 3. Why version 1 was still the right first step

Even though version 1 was single-user specific, it was exactly the right first version.

Why:

- the proctoring rules needed to be proven before scaling
- scoring logic needed to be understood and tuned
- false positives needed to be seen visually
- evidence capture needed to be validated
- behavior combinations such as phone plus looking down needed to be tested end to end

If we had tried to build the multi-user service first, we would have mixed two different problems:

- "Is the proctoring logic correct?"
- "Can the service architecture scale?"

Version 1 solved the first problem cleanly.

## 4. Restrictions that became visible after version 1

Once version 1 proved the core pipeline, the next set of problems became obvious.

### 4.1 One candidate per machine/runtime

The system could monitor one person well, but only through one local runtime. That is not enough for a real exam platform.

### 4.2 No browser-native candidate flow

Real users would join through a browser, not through a local Python/OpenCV app. Version 1 had no direct story for:

- browser permissions
- browser media streaming
- tab-switch tracking
- focus-loss detection

### 4.3 No central backend control for many candidates

There was no place where multiple users could be observed together. That meant:

- no active session list
- no centralized alert visibility
- no per-candidate drill-down pages
- no shared reporting workflow

### 4.4 No efficient shared inference

If version 1 had been copied once per user, we would have created:

- duplicate model loading
- duplicate loops
- inefficient CPU usage
- poor coordination

That would not be a clean multi-user architecture. It would just be many isolated single-user processes.

### 4.5 Time behavior depended too much on local FPS

Some temporal logic in version 1, especially object smoothing, was still frame-based. That is acceptable when one local camera has roughly stable performance, but it becomes unreliable across different users and network conditions.

### 4.6 Rich local proof was expensive

The proof model in version 1 was very useful, but heavy evidence generation per event is harder to support once many candidates are active simultaneously.

These restrictions did not mean version 1 was wrong. They meant the next version needed a different architecture, not just more optimization.

## 5. The design goal of version 2

Version 2 was built to answer the next question:

"How do we keep the proctoring intelligence from version 1, but reorganize the system so one backend can manage multiple remote candidates through the browser?"

That required changing the shape of the system in several fundamental ways.

## 6. What changed in version 2 so the system could handle multiple users

Version 2 did not simply add "more users." It introduced the missing architectural layers that multi-user proctoring needs.

### 6.1 The candidate became a browser client

This was one of the biggest changes.

Instead of:

- local OpenCV camera capture
- local microphone capture
- local Python UI

version 2 moved the candidate experience into a Next.js page that:

- asks for camera and microphone permission
- creates an `RTCPeerConnection`
- streams audio and video to the backend
- listens for warnings and alerts through SSE
- reports tab switches and focus loss

Why this mattered:

- candidates can join remotely
- the backend can remain the centralized decision point
- proctoring becomes product-shaped rather than prototype-shaped

### 6.2 A real media transport layer was introduced

WebRTC solved the biggest gap in version 1: how remote media reaches the backend in real time.

This change added:

- SDP offer/answer handling
- peer connection lifecycle management
- track receivers for audio and video
- browser-native streaming

This was essential because multi-user support is impossible without a transport layer that can carry multiple remote streams into a central service.

### 6.3 Session became a first-class concept

Version 2 introduced a dedicated `ProctorSession` per candidate.

That means every candidate now has their own:

- behavioral history
- risk engine state
- event log
- proof output
- SSE subscribers
- cleanup lifecycle

This is one of the most important architectural upgrades in the whole project.

Why:

- the system can now reason about many candidates independently
- one candidate's risk timeline does not interfere with another's
- monitoring, reports, and event streaming become structured

### 6.4 A central coordinator was added

Version 2 introduced `ProctorCoordinator`, which acts as the shared orchestration layer for all active sessions.

This directly addressed a major version 1 restriction.

Instead of every candidate having a separate isolated detector loop, the backend now:

- collects the latest frames from multiple sessions
- runs YOLO across them in a coordinated way
- dispatches results back to the correct session
- advances all sessions on a controlled tick

Why this mattered:

- resource use became more organized
- batch-style processing became possible
- the backend gained a single place to control scheduling

This is a foundational multi-user change.

### 6.5 Shared inference replaced isolated local inference

In version 1, the whole pipeline existed for one user only. In version 2, some heavy work was centralized and shared.

Most importantly:

- object detection became shared at the coordinator level

That means the system no longer behaves like "many copies of version 1." It behaves like one coordinated service running many sessions.

### 6.6 Face processing became more reusable

Version 2 introduced a shared `FaceMeshProvider` inside each session so multiple face-dependent components could use the same landmark result.

This addressed an important scale issue:

- repeating landmark work for every face-related detector is expensive
- multi-user CPU systems need to avoid duplicate work aggressively

So version 2 did not only add concurrency. It also reduced wasted inference inside each session.

### 6.7 Temporal logic became more suitable for unstable frame rates

Version 1 used frame-oriented smoothing in places. That made sense for a local loop. Version 2 changed object tracking to a time-based model.

Why this was necessary:

- different remote users can have different effective FPS
- network conditions can vary
- browser and device performance can vary

A time-based window preserves policy meaning better than a frame-count window in a multi-user remote setting.

This is a subtle but very important scaling improvement.

### 6.8 Browser-only cheating signals were added

Version 1 could infer visual and audio behavior, but it could not know when a user switched tabs because it had no browser layer.

Version 2 solved that by adding frontend reporting for:

- `visibilitychange`
- window blur and focus transitions

This gave the backend a new type of evidence that is only possible once the system has a browser client and a server API.

### 6.9 Centralized admin visibility was introduced

Version 2 added:

- admin dashboard
- session detail monitoring
- snapshot viewing
- reports pages

This addressed another version 1 limitation: the lack of a central operational view.

Now the system could support:

- multiple live candidates
- one admin monitoring many sessions
- historical report review after session completion

### 6.10 Proof generation was made lighter for scale

Version 1 had richer proof generation because it was a single-user prototype. Version 2 simplified proof into lighter artifacts such as:

- JPEGs for visual alerts
- WAV for speaker-audio evidence

Why this was changed:

- multi-user CPU architecture cannot spend the same overhead on every event
- evidence still matters, but proof must fit shared-resource constraints

This is a classic sign of progress from prototype to service.

## 7. What version 2 did not change

It is also important to say what stayed intentionally stable.

Version 2 did not throw away the core proctoring logic from version 1.

The system still relied on:

- object detection
- face and gaze analysis
- liveness logic
- mouth and audio fusion
- temporal persistence checks
- fixed and decaying risk
- warning, alert, and termination policy
- suspicious combination scoring

That continuity is important because version 2 was not a restart. It was a re-architecture of the same intelligence into a scalable service shape.

## 8. In simple terms: how version 1 became version 2

The easiest way to understand the progress is this:

In version 1, the system was built like a powerful local examiner for one candidate.

In version 2, the system was rebuilt as a central exam-control platform where:

- each candidate is a remote session
- the browser sends media to the backend
- the backend runs shared scheduling and detection
- each session keeps its own behavior history and risk state
- admins can observe many sessions centrally

So the move from version 1 to version 2 was not mainly about changing the cheating rules. It was about changing the delivery model and execution model of those rules.

## 9. Restrictions that still remained after version 2

Version 2 solved the single-user limitation, but it exposed the next level of constraints.

Main remaining issues:

- CPU remained the main bottleneck
- concurrent user count was still limited
- WebRTC needed more deployment hardening
- cloud networking realities introduced new edge cases
- runtime tuning was still more limited than ideal
- multi-instance operation was only partially prepared

Those limits are exactly why version 3 exists.

## 10. How version 3 continues the same progression

Version 3 should be understood as the next natural step after version 2.

If version 2 solved:

- "How do we go from one local candidate to many browser-based candidates?"

then version 3 solves:

- "How do we run that multi-user browser-based system more reliably and efficiently in real deployment conditions?"

So version 3 introduces:

- GPU-first YOLO inference
- Docker packaging
- AWS-ready deployment assumptions
- trickle ICE and stronger WebRTC handling
- runtime admin settings
- load testing and benchmarks
- multi-backend-aware frontend support

This means the progression across versions is very coherent:

- version 1 proved detection and scoring
- version 2 proved centralized multi-user architecture
- version 3 proved deployable and tunable multi-user infrastructure

## 11. Final interpretation of the progress

The real progress of the project is not just "more features in each version." The real progress is that each version solved the most important problem exposed by the previous one.

Version 1 solved:

- can the proctoring intelligence work at all?

Version 2 solved:

- can that intelligence be reorganized to support multiple remote users?

Version 3 solved:

- can that multi-user system be accelerated, hardened, tuned, and deployed more realistically?

That is why the versions feel like natural iterations rather than disconnected implementations. Each one keeps the valuable parts of the previous version, removes the biggest current restriction, and makes the system ready for the next scale of use.
