# Version 1 Architecture

Source implementation: `D:\VS Code\POC\ai-proctor-vision-poc`

This document explains the first implementation in depth. This version is the single-user, CPU-oriented proof of concept. It is the most direct expression of the full proctoring logic: one local webcam, one local microphone, one inference loop, one risk engine, one alert pipeline, and one proof pipeline. Later versions keep most of the detection ideas but reorganize transport, scheduling, and resource sharing for scale.

## 2. Runtime shape and top-level architecture

The entry point is [`main.py`](D:\VS Code\POC\ai-proctor-vision-poc\main.py). The application is a local OpenCV loop with background audio capture.

High-level component layout:

- `cv2.VideoCapture(0)` reads frames from the local camera.
- `ObjectDetector` runs YOLO object detection on each frame.
- `HeadPoseDetector` uses MediaPipe FaceMesh landmarks to estimate face visibility, yaw, pitch, gaze, blink state, and partial-face conditions.
- `LivenessDetector` uses temporal variation in face pose and blink behavior to detect fake presence.
- `LipDetector` estimates mouth aspect ratio, speaking state, and yawning.
- `AudioMonitor` captures microphone audio continuously and runs VAD in a background thread.
- `SpeakerAudioDetector` compares audio speech activity with lip activity to detect external speaker audio.
- `ObjectTemporalTracker` smooths short-lived detection flicker.
- `RiskEngine` converts observed events into risk points and termination conditions.
- `AlertEngine` decides whether to produce a warning, an alert, or a terminal session-end event.
- `ProofWriter` saves images, short clips, and optionally audio-backed evidence around important events.
- `AlertManager` stores short-lived warning and alert messages for on-screen rendering.

This architecture is intentionally simple: one synchronous video loop for visual inference, plus one asynchronous audio subsystem. That simplicity makes it ideal for understanding the full behavior of the system because almost all state transitions happen in one place and in time order.

## 3. End-to-end processing flow

The frame pipeline inside [`main.py`](D:\VS Code\POC\ai-proctor-vision-poc\main.py) is:

1. Read one webcam frame.
2. Run YOLO object detection on the frame.
3. Merge selected classes into stable semantic flags.
4. Run head pose and gaze analysis from FaceMesh.
5. Run liveness analysis using recent head motion and blink history.
6. Run lip analysis for speaking and yawning.
7. Read current microphone speech state from the audio monitor.
8. Infer "speaker audio" when speech exists without matching visible lip movement.
9. Smooth object presence through the temporal tracker.
10. Convert all current behaviors into event labels.
11. Feed events into the risk engine.
12. Ask the alert engine whether to emit a warning, alert, or termination.
13. Save proof when configured.
14. Draw overlay information on the frame.
15. Show the frame locally with `cv2.imshow`.

The logic is intentionally ordered this way:

- object detection happens first because person count, phone, book, and earbud presence inform later decisions
- face analysis happens before speaker-audio logic because visible lip motion is needed to decide whether voice belongs to the candidate
- temporal smoothing happens before scoring because the system wants risk based on persistent behavior, not single-frame noise
- risk generation happens before alert emission because alerts should be a consequence of scored policy, not raw detections

## 4. Visual detection subsystem

### 4.1 Object detection

The object detector is implemented in [`detectors/object_detector.py`](D:\VS Code\POC\ai-proctor-vision-poc\detectors\object_detector.py). It uses a YOLO model to detect cheating-relevant classes.

Observed class handling in this version:

- `person`
- `cell phone`
- `book`
- `earbud`
- `headphones`

Important confidence thresholds from [`config.py`](D:\VS Code\POC\ai-proctor-vision-poc\config.py):

- default object confidence: `0.5`
- person confidence: `0.3`
- phone confidence: `0.65`
- book confidence: `0.70`
- audio-device confidence: `0.41`

### 4.2 Person-count normalization

`main.py` consolidates YOLO outputs into simple booleans and counts used by the rest of the system:

- whether at least one person exists
- whether more than one person exists
- whether a phone exists
- whether a book exists
- whether an earbud exists

This is important because the risk system is event based, not raw-box based. The engine does not care about every rectangle; it cares about normalized behaviors such as "multiple people present continuously."

## 5. Face, head, and gaze subsystem

### 5.1 Landmark extraction

[`detectors/head_pose_detector.py`](D:\VS Code\POC\ai-proctor-vision-poc\detectors\head_pose_detector.py) uses MediaPipe FaceMesh with refined landmarks enabled. FaceMesh provides dense face landmarks that are then converted into interpretable posture features.

The detector estimates:

- face presence
- face size adequacy
- partial face visibility
- yaw
- pitch
- gaze left or right
- blink state through EAR

Important thresholds from [`config.py`](D:\VS Code\POC\ai-proctor-vision-poc\config.py):

- minimum usable face width: `80`
- minimum usable face height: `95`
- yaw threshold: `0.20`
- looking down threshold: `0.13`
- looking up threshold: `-0.10`
- gaze left threshold: `-0.13`
- gaze right threshold: `0.13`
- eye-aspect-ratio threshold: `0.20`
- blink minimum frames: `2`

Why these features matter:

- yaw and gaze capture side-looking behavior that often corresponds to looking at another screen or person
- pitch captures looking down at a phone, notes, or desk material
- face-size thresholds protect the system from scoring very distant or partial detections as if they were reliable measurements
- blink state is reused by liveness detection

### 5.2 Head-direction semantics

The detector does not just expose raw numbers. It maps them into semantic states such as:

- looking left
- looking right
- looking down
- looking away
- face hidden
- partial face

This abstraction is critical because the scoring layer works in policy language rather than geometry language.

### 5.3 Temporal head tracking

[`core/head_tracker.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\head_tracker.py) tracks how long suspicious head states persist. In version 1, the tracker primarily yields whether a state is currently active, while duration is inferred externally from state timestamps.

Important time thresholds from [`config.py`](D:\VS Code\POC\ai-proctor-vision-poc\config.py):

- looking away threshold: `2.0s`
- gaze threshold: `1.5s`

Why duration gating exists:

- a single glance is common and should not behave like strong evidence
- persistent off-screen attention is more meaningful than transient motion
- duration gating dramatically reduces false positives from normal reading, blinking, or posture adjustment

## 6. Liveness subsystem

[`core/liveness.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\liveness.py) tries to detect fake presence, such as a frozen screen, static photo, or unnaturally motionless substitute.

The liveness logic combines:

- temporal variance of yaw
- temporal variance of gaze
- temporal variance of pitch
- blink recency

Important liveness parameters:

- sample interval: `0.2s`
- fake window: `15s`
- minimum motion variance: `0.001`
- no-blink timeout: `10s`
- weighting: yaw `0.45`, gaze `0.45`, pitch `0.10`

Interpretation:

- if the face remains too stable across the recent window and blink behavior is absent or implausible, the system treats the presence as suspicious
- yaw and gaze dominate because they usually vary subtly even when a real person tries to stay still
- pitch contributes less because many users keep similar head height for long periods

Why liveness is separate from "no person":

- `no_person` means nobody is visible
- `fake_presence` means something person-like is visible but behaves unlike a live candidate

That distinction matters because the policy impact is different.

## 7. Mouth and audio subsystem

### 7.1 Lip analysis

[`detectors/lip_detector.py`](D:\VS Code\POC\ai-proctor-vision-poc\detectors\lip_detector.py) uses FaceMesh landmarks to derive mouth aspect ratio and classify:

- speaking-like lip motion
- yawning

Important thresholds:

- speaking threshold: `0.05`
- yawn threshold: `0.22`
- yawn minimum duration: `1.5s`
- dynamic speaking standard-deviation threshold: `0.010`
- mouth history size: `30`

Why mouth analysis is needed:

- the system must distinguish the candidate speaking from external speech in the room
- mouth activity also adds behavioral context for oral communication and fatigue cues

### 7.2 Audio capture

[`core/audio_monitor.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\audio_monitor.py) captures microphone audio on a background thread. It uses:

- mono audio
- `16 kHz` sample rate
- chunk size `512`
- Silero VAD for speech detection

Important thresholds:

- speech probability threshold: `0.5`
- speaker hold time: `0.3s`

The audio monitor maintains recent speech state independently of the frame loop so that video processing jitter does not block audio observation.

### 7.3 Speaker-audio inference

The system uses `SpeakerAudioDetector` to infer suspicious room speech:

- if speech is detected but lips are not moving
- or speech is detected with no valid face context

then the system treats it as external speaker audio rather than candidate speech.

This is a clever fusion rule because audio alone cannot tell who is speaking. The system uses visual evidence to attribute speech.

## 8. Temporal smoothing and stability

[`core/object_tracker.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\object_tracker.py) is a frame-count-based temporal tracker. It keeps recent boolean observations in a sliding deque and only promotes an object to active once enough votes accumulate.

Important tracker parameters:

- window size: `15` frames
- minimum votes: `5`
- phone votes: `9`
- book votes: `10`
- earbud votes: `9`

Why this exists:

- YOLO detections flicker
- small objects like earbuds are unstable
- scoring should depend on persistence, not on a single lucky or unlucky frame

This first implementation is frame based, which is acceptable for a single local loop. Later versions move to time-based windows because frame-based windows become inconsistent when FPS varies between users.

## 9. Risk model and policy engine

[`core/risk_engine.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\risk_engine.py) is the heart of the proctoring policy. It does not score raw detections directly; it scores policy-level suspicious events.

### 9.1 Two risk buckets

The engine keeps two types of risk:

- fixed risk
- decaying risk

Fixed risk is for serious evidence that should remain attached to the session record.

Examples:

- repeated phone usage
- multiple people
- no person
- fake presence

Decaying risk is for lower-certainty or short-lived suspicious behavior that should fade if the candidate returns to normal.

Examples:

- gaze events
- looking down briefly
- face partially hidden
- book presence
- headphones

This split is one of the strongest design choices in the whole system because it avoids both extremes:

- pure permanent scoring punishes a candidate forever for small transient issues
- pure decaying scoring forgets strong evidence too quickly

### 9.2 Session thresholds

Risk thresholds from the engine:

- warning threshold: `30`
- high-risk threshold: `60`
- admin-review threshold: `100`

Decay behavior:

- decay interval is `max(60, session_duration / 20)`
- with session duration `3600s`, decay runs every `180s`
- each decay step removes `5` from the decaying bucket

Session duration parameter:

- `RISK_SESSION_DURATION_S = 3600`

Why session-length-aware decay exists:

- a 3-hour exam and a 10-minute exam should not decay suspicious activity on the same cadence
- scaling decay with session length makes the model more stable across exam formats

### 9.3 Non-decaying event categories

The engine explicitly treats these as non-decaying:

- `tab_switch`
- `phone`
- `fake_presence`
- `multiple_people`
- `no_person`

Even though `tab_switch` is listed in the engine, version 1 does not have the browser-based tab-switch reporting path yet. The category exists as part of the evolving design.

### 9.4 Event scoring rules

Observed policy behavior in this version:

- `phone`
  - first occurrence gets a grace warning
  - second occurrence adds `25 * confidence` fixed risk
  - third and later occurrences add `50 * confidence` fixed risk
- `headphone` or `earbud`
  - first occurrence gets grace
  - later occurrences add `20 * confidence` decaying risk
- `multiple_people`
  - first occurrence gets grace
  - second occurrence adds `20` fixed risk
  - third and later occurrences add `50` fixed risk
  - continuous presence for `20s` terminates the session
- `book`
  - adds `20 * confidence` decaying risk
- `fake_presence`
  - at `>=10s` adds `30` fixed risk
  - at `>=25s` adds `60` fixed risk
- `face_hidden`
  - at `>=5s` adds `10` decaying risk
  - at `>=10s` adds `20` decaying risk
- gaze-style events
  - each qualifying event adds `5` decaying risk, subject to cooldowns
- `partial_face`
  - after `5s`, adds `2` decaying risk
- `no_person`
  - at `>=5s` adds `25` fixed risk
  - at `>=10s` adds `50` fixed risk
  - continuous absence for `20s` terminates the session
- `speaker_audio`
  - first speaking episode shorter than `3s` generates warning only
  - if it continues, it adds `10` decaying risk
  - then repeated `20` fixed-risk increments can occur every `10s`
  - later speaking episodes use a longer `5s` grace before escalation

### 9.5 Combination logic

The engine also awards extra risk when suspicious signals align:

- looking down plus book presence adds `15` decaying risk
- phone plus looking down adds `20` decaying risk with a `60s` cooldown
- three gaze scoring events within `30s` add an extra `10` decaying risk

Why combinations matter:

- an isolated cue can be ambiguous
- a coordinated pattern is stronger evidence

For example, looking down alone may be harmless, but looking down while a book is stably detected is much more suspicious.

## 10. Warning, alert, and termination procedure

[`core/alert_engine.py`](D:\VS Code\POC\ai-proctor-vision-poc\core\alert_engine.py) converts scoring outcomes into user-facing or admin-facing events.

Operational pattern:

- if an event produced no risk but is still policy-relevant, the engine may emit a warning
- if an event produced risk, the engine emits an alert subject to API cooldown
- if a termination condition is reached, it emits a single terminal alert

This matters because not every suspicious action should immediately look like a severe violation. The system uses warnings as a soft intervention and alerts as a formally scored escalation.

Display handling is then managed by [`utils/alerts.py`](D:\VS Code\POC\ai-proctor-vision-poc\utils\alerts.py):

- warnings are shown briefly, around `3s`
- alerts are shown longer, around `5s`

Why warnings exist:

- they give the candidate a chance to self-correct
- they reduce unnecessary punishment for accidental behavior
- they let operators validate detector quality during prototyping

## 11. Proof and evidence generation

[`utils/proof_writer.py`](D:\VS Code\POC\ai-proctor-vision-poc\utils\proof_writer.py) is much richer in version 1 than in later CPU multi-user builds.

Capabilities:

- save still proof images
- save short proof videos
- optionally combine audio and video when supported
- preserve a ring buffer of frames before the event
- include a few seconds after the event

Important proof settings:

- `SAVE_REPORT = True`
- `SAVE_PROOF = True`
- pre-event window: `2.5s`
- post-event window: `2.5s`
- proof FPS: `20`

Why this is powerful in the POC:

- it makes debugging far easier because developers can inspect exactly what triggered the policy
- it supports threshold tuning
- it creates evidence packages for manual review

Why later versions simplify it:

- rich clip generation is expensive for multi-user CPU deployments
- synchronous proof generation can become a bottleneck at scale

## 12. Why this architecture was chosen

This first version makes several deliberate tradeoffs.

### 12.1 Single-process simplicity

All major decisions happen locally and mostly synchronously. That reduces debugging complexity and makes the event timeline easy to reason about.

### 12.2 CPU-first design

The system can run on commodity machines for experimentation. That is valuable when validating policy logic before investing in GPU infrastructure.

### 12.3 Explicit rule engine

Instead of a black-box cheating score model, the system uses a transparent rules engine. This makes it easier to:

- audit why risk increased
- tune thresholds
- explain behavior to operators
- add new policies without retraining

### 12.4 Rich evidence capture

Because this version is a proof of concept, it prioritizes observability over throughput.

## 13. Limitations of version 1

The first implementation is intentionally not production scale.

Main limitations:

- one local user only
- local camera and microphone assumptions
- no browser-native candidate flow
- no centralized session management
- no shared inference batching
- repeated MediaPipe usage across components adds CPU load
- frame-count-based smoothing depends on local FPS stability

These limitations directly motivate version 2.

## 14. What carries forward into later versions

Even though the system architecture changes in later iterations, the core ideas from version 1 remain foundational:

- cheating signals should be fused across modalities
- persistent behavior matters more than single-frame detections
- scoring should separate fixed evidence from decaying suspicion
- warnings should exist before heavy escalation where appropriate
- proof must be tied to scored events
- combination logic is stronger than isolated cues

In that sense, version 1 is not just a prototype. It is the policy and signal-processing blueprint for the later multi-user systems.
