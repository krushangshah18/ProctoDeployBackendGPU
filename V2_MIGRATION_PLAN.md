# ProctorPod v2 — MediaPipe → InsightFace + L2CS-Net Migration Plan

> **Purpose**: Reference document for implementing v2 in a fresh repo clone.
> Do NOT apply these changes to the current repo — this is the blueprint only.
> Read `SESSION_CONTEXT.md` and `CLAUDE.md` first for full system context.
> Last updated: 2026-03-23

---

## 1. Why This Change

### The core problem

MediaPipe FaceMesh **holds the Python GIL** during inference. This means that even though we
use a `ThreadPoolExecutor`, all MediaPipe calls execute **serially** — not in parallel.

The scaling math:

```
Tick budget = 100ms (10 Hz target)

Current (MediaPipe CPU, GIL held, stride=3):
  ceil(N/3) sessions get fresh FaceMesh per tick
  Each call ≈ 15ms, all serial

  5  users → ceil(5/3)=2  sessions × 15ms = 30ms  → hidden behind YOLO's ~60ms ✅
  10 users → ceil(10/3)=4 sessions × 15ms = 60ms  → equals YOLO, no hiding    ⚠️
  15 users → ceil(15/3)=5 sessions × 15ms = 75ms  → EXCEEDS YOLO, MP limits   ❌
```

The ceiling at ~9-10 users is a **software limit**, not hardware. The T4 GPU and 4 vCPUs
are underutilised — MediaPipe is the bottleneck.

### Why InsightFace + L2CS-Net solves it

- **InsightFace** runs via **ONNX Runtime with CUDA execution provider**.
  `session.run()` releases the GIL during GPU inference → true concurrent execution
  in the ThreadPoolExecutor (each thread dispatches to GPU independently).

- **L2CS-Net** is a PyTorch gaze model running on CUDA.
  PyTorch CUDA ops also release the GIL.

- Both models are GPU-accelerated: per-session inference drops from ~15ms (CPU) to ~3-5ms (GPU).

New scaling math (stride=3, InsightFace ~4ms/session):

```
5  users → ceil(5/3)=2  × 4ms = 8ms   — hidden behind YOLO ✅
10 users → ceil(10/3)=4 × 4ms = 16ms  — hidden behind YOLO ✅
15 users → ceil(15/3)=5 × 4ms = 20ms  — still hidden        ✅
20 users → ceil(20/3)=7 × 4ms = 28ms  — still hidden        ✅
```

This unlocks 3-4× more concurrent users on the same g4dn.xlarge hardware.

---

## 2. Feature Preservation Map

Every feature currently using MediaPipe must be preserved. This table shows the source
of each signal in v1 vs v2:

| Feature | v1 Source | v2 Source | Notes |
|---------|-----------|-----------|-------|
| Head yaw (looking away) | MediaPipe PnP solve | InsightFace `face.pose` | Built-in, more accurate |
| Head pitch (down/up) | MediaPipe PnP solve | InsightFace `face.pose` | Built-in |
| Gaze left/right | MediaPipe iris landmarks 468, 473 | L2CS-Net gaze estimator | Dedicated model, ~same accuracy |
| EAR / blink detection | MediaPipe eye landmarks [33,160,158,133,153,144] | InsightFace 106-pt eye region | 8 eye pts per eye — sufficient for EAR |
| Lip MAR / speaking | MediaPipe landmarks 13,14,78,308 | InsightFace 106-pt lip region | Inner lip landmarks available |
| Lip yawn duration | Same MAR logic | Same MAR logic — different indices only | No logic change |
| Partial face detection | MediaPipe face bbox | InsightFace `face.bbox` | Direct equivalent |
| Face hidden (no landmarks) | `results.multi_face_landmarks` is None | `app.get(frame)` returns empty list | Direct equivalent |
| Fake presence (liveness) | Landmark variance over time | Same — landmark variance from IF points | Same logic, different indices |

**Nothing is removed.** Gaze detection (`DETECT_LOOKING_SIDE`) changes source from
MediaPipe iris to L2CS-Net — detection quality is equivalent or better.

---

## 3. New Models

### 3.1 InsightFace — buffalo_l pack

InsightFace provides several model packs. Use `buffalo_l` (large, most accurate):

```
buffalo_l contains:
  det_10g.onnx         ← face detection (RetinaFace-style)
  2d106det.onnx        ← 106-point 2D landmark model
  1k3d68.onnx          ← 68-point 3D landmark + head pose (use this for pose)
  glintr100.onnx       ← face recognition (NOT needed for proctoring)
  w600k_r50.onnx       ← face recognition (NOT needed for proctoring)
```

For proctoring we need: **det_10g + 1k3d68 + 2d106det**. Skip the recognition models
to save VRAM and inference time — InsightFace supports selective model loading.

**Head pose**: `1k3d68.onnx` outputs the 68 3D landmarks AND the pose matrix
(pitch/yaw/roll in degrees). This is `face.pose` on the result object.

**Landmarks**: Use `2d106det.onnx` for 2D landmarks (EAR, lip MAR, face size check).

### 3.2 L2CS-Net — gaze estimation

- Repository: `https://github.com/Ahmednull/L2CS-Net`
- Pip package: `l2cs` (wraps the pretrained model download)
- Pre-trained weights: available from the repo (MPIIFaceGaze or ETH-XGaze variants)
- Input: face crop BGR (224×224), or full frame with face bbox
- Output: `(pitch_rad, yaw_rad)` — gaze direction angles in radians
- Hardware: PyTorch CUDA, runs on same T4 GPU as YOLO
- Latency: ~3–5ms per face crop on T4

**Mapping to current gaze thresholds**:

```python
# Current (MediaPipe) — normalised ratios relative to eye width
GAZE_LEFT  = -0.13   # iris center is 13% of eye-width to the left
GAZE_RIGHT =  0.13

# New (L2CS-Net) — yaw angle in radians
# Rough equivalence: 0.13 ratio ≈ 8-10 degrees ≈ 0.14-0.17 radians
# Calibrate empirically after integration — start with:
GAZE_LEFT  = -0.15   # radians
GAZE_RIGHT =  0.15   # radians
```

The thresholds are configurable in `config.py` so calibration is a one-line change.

---

## 4. InsightFace 106-Point Landmark Layout

The `2d106det` model returns 106 landmarks (x, y) in pixel coordinates.
Key regions for v2:

```
Face contour : indices 0–32    (33 points, jawline)
Left eyebrow : indices 33–42   (10 points)
Right eyebrow: indices 43–52   (10 points)
Nose bridge  : indices 53–58   (6 points)
Nose tip     : indices 59–67   (9 points)
Left eye     : indices 68–75   (8 points) ← use for EAR
Right eye    : indices 76–83   (8 points) ← use for EAR
Outer lip    : indices 84–95   (12 points)
Inner lip    : indices 96–103  (8 points) ← use for MAR
```

### EAR with InsightFace eye landmarks

InsightFace left eye (8 points, indices 68–75):
```
68 = left corner
69 = top-left
70 = top-center-left
71 = top-center-right
72 = right corner
73 = bottom-center-right
74 = bottom-center-left
75 = bottom-left
```

EAR formula (same as current — 6-point reduction from 8):
```python
# Pick 6 representative points from the 8 IF eye landmarks
LEFT_EYE_EAR_IDX  = [68, 69, 71, 72, 74, 75]   # left/right, 2×top, 2×bottom
RIGHT_EYE_EAR_IDX = [76, 77, 79, 80, 82, 83]

def ear(pts):
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)
```

### Lip MAR with InsightFace inner lip landmarks

Inner lip (indices 96–103, 8 points):
```
96  = top-center
100 = bottom-center
98  = left corner
102 = right corner  (approximate — verify empirically)
```

MAR formula (unchanged from v1):
```python
TOP   = landmarks[96]
BOT   = landmarks[100]
LEFT  = landmarks[98]
RIGHT = landmarks[102]
mar   = dist(TOP, BOT) / (dist(LEFT, RIGHT) + 1e-6)
```

### Head pose with InsightFace 1k3d68

The `1k3d68` model outputs a `pose` attribute directly:
```python
face = app.get(frame)[0]
yaw   = face.pose[1]   # degrees, positive = right
pitch = face.pose[0]   # degrees, positive = down
roll  = face.pose[2]
```

**Important**: Current code uses normalised ratios (not degrees). Must normalise
InsightFace degrees to the same [-1, +1] range for the existing threshold comparisons,
OR update `LOOK_AWAY_YAW`, `LOOK_DOWN_PITCH`, `LOOK_UP_PITCH` in `config.py` to degree
values. The latter is cleaner — just calibrate the config thresholds.

Approximate equivalence:
```
LOOK_AWAY_YAW   = 0.20  (ratio) ≈ 20–25 degrees
LOOK_DOWN_PITCH = 0.13  (ratio) ≈ 12–15 degrees
LOOK_UP_PITCH   = -0.10 (ratio) ≈ -8 to -12 degrees
```

Suggested starting config values (verify empirically):
```python
LOOK_AWAY_YAW   = 22    # degrees
LOOK_DOWN_PITCH = 14    # degrees
LOOK_UP_PITCH   = -10   # degrees
```

---

## 5. File-by-File Changes

### 5.1 NEW: `detectors/insightface_provider.py`

Replaces `detectors/face_mesh_provider.py`.

Responsibilities:
- Load InsightFace `buffalo_l` pack, **excluding recognition models** (saves VRAM + time)
- Expose `process(frame) → face | None` where `face` is the InsightFace result object
- Run on CUDA via ONNX Runtime (`ctx_id=0`)
- One instance per session (same pattern as `FaceMeshProvider`)

Key points:
```python
import insightface
from insightface.app import FaceAnalysis

class InsightFaceProvider:
    def __init__(self, ctx_id: int = 0, det_size: tuple = (320, 320)):
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"],
            # excludes: recognition, genderage — saves ~200ms init and VRAM
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process(self, frame: np.ndarray):
        # Returns first detected face or None
        faces = self._app.get(frame)
        return faces[0] if faces else None

    def close(self) -> None:
        pass  # ONNX Runtime sessions cleaned up by GC
```

**`det_size=(320, 320)`**: Face detection input size. Smaller = faster detection.
320×320 is sufficient for a webcam frame (faces are large relative to frame).
Can try 160×160 if detection speed is critical — but may miss small/distant faces.

**Thread safety**: ONNX Runtime sessions are NOT thread-safe by default.
Each `InsightFaceProvider` instance must be used by **one thread only** — same model
as `FaceMeshProvider`. Since each session owns its own provider instance, this is safe.

### 5.2 NEW: `detectors/gaze_estimator.py`

Wraps L2CS-Net for per-session gaze estimation.

Responsibilities:
- Load L2CS-Net model once (shared across calls — stateless)
- Accept face bbox + full frame → crop face → run inference → return `(pitch, yaw)` radians
- Run on CUDA via PyTorch

Key points:
```python
from l2cs import Pipeline as L2CSPipeline
import torch

class GazeEstimator:
    def __init__(self, device: str = "cuda"):
        self._pipeline = L2CSPipeline(
            weights=CWD / "models" / "L2CSNet_gaze360.pkl",
            device=torch.device(device),
        )

    def estimate(self, frame: np.ndarray, bbox) -> tuple[float, float] | None:
        """
        Returns (pitch_rad, yaw_rad) or None if estimation fails.
        bbox: (x1, y1, x2, y2) in pixel coords from InsightFace face.bbox
        """
        ...
```

**Important**: L2CS-Net pretrained weights need to be bundled in the Docker image.
Add a download step in the Dockerfile or commit the weights file (it's ~100 MB).
Check the L2CS-Net repo for the best pretrained checkpoint (Gaze360 dataset preferred
over MPIIFaceGaze for general gaze — broader training distribution).

**Shared vs per-session**: L2CS-Net is stateless (no tracking state). Can be a
**coordinator-level singleton** shared across all sessions, unlike `InsightFaceProvider`
which maintains per-session state. This saves VRAM and init time.

### 5.3 MODIFIED: `detectors/head_pose_detector.py`

Heavy rework — remove all MediaPipe references, use InsightFace + L2CS-Net.

Changes:
- Remove `import mediapipe as mp`
- Remove `own_mesh` parameter and internal `face_mesh` creation
- Remove PnP solve (head pose comes from InsightFace `face.pose` directly)
- Remove iris landmark indices (468, 473) — gaze comes from `GazeEstimator`
- Update `detect()` signature: accepts `face` (InsightFace result) instead of `landmarks`
- Update eye landmark indices to InsightFace 106-pt scheme (68–75 left, 76–83 right)
- Update draw_debug to use InsightFace landmark pixel coords

Return tuple is unchanged (12-element):
```python
(looking_away, looking_down, looking_up,
 looking_left, looking_right,
 partial_face,
 yaw, pitch, gaze,
 ear, blinked, total_blinks)
```

Yaw/pitch are now in **degrees** (from `face.pose`) — update config thresholds accordingly.
Gaze is now from L2CS-Net yaw output in radians — update `GAZE_LEFT`/`GAZE_RIGHT`.

### 5.4 MODIFIED: `detectors/lip_detector.py`

Moderate rework — update landmark indices, remove MediaPipe.

Changes:
- Remove `import mediapipe as mp`
- Remove `own_mesh` parameter
- Update `process()` signature: accepts `face` (InsightFace result) instead of `landmarks`
- Replace `_TOP=13, _BOT=14, _LEFT=78, _RIGHT=308` with InsightFace inner lip indices
- Replace `_LIP_OUTLINE` indices for debug drawing
- Extract `(x, y)` pixel coords from `face.landmark_2d_106` array (numpy array, shape 106×2)

`LipState` dataclass is unchanged — same return contract to all callers.

### 5.5 MODIFIED: `detectors/__init__.py`

Update exports:
```python
# Remove:
from detectors.face_mesh_provider import FaceMeshProvider

# Add:
from detectors.insightface_provider import InsightFaceProvider
from detectors.gaze_estimator import GazeEstimator
```

### 5.6 MODIFIED: `core/proctor_session.py`

Changes:
- Replace `FaceMeshProvider` with `InsightFaceProvider`
- Add `GazeEstimator` (if not coordinator-level singleton — see option below)
- Update `run_mediapipe()` → rename to `run_insightface()` (or keep name for minimal diff)
- Pass `face` object to `head_detector.detect()` and `lip_detector.process()` instead of `landmarks`
- Remove `_FakeLandmark` class (was leftover from ProcessPool experiment — no longer needed)
- Remove `_last_lm_bytes` attribute (same reason)

`run_mediapipe()` new body:
```python
def run_mediapipe(self, frame: np.ndarray) -> tuple:
    """Name kept for coordinator compatibility."""
    from detectors.lip_detector import LipState
    if not self.needs_mediapipe():
        return (_NULL_HEAD, LipState(face_detected=False))

    face = self._if_provider.process(frame)
    ts   = time.monotonic() - self.session_clock

    head_result = self.head_detector.detect(frame, face=face, draw=False)
    lip_result  = self.lip_detector.process(frame, ts, face=face, draw=False)

    self._last_mp_result = (head_result, lip_result)
    return head_result, lip_result
```

**GazeEstimator placement options**:

Option A — Per-session (simpler, more VRAM):
```python
# in __init__:
self._gaze_estimator = GazeEstimator(device="cuda")
# passed to HeadPoseDetector.__init__ or called directly in run_mediapipe
```

Option B — Coordinator-level singleton (less VRAM, shared):
```python
# in ProctorCoordinator.__init__:
self.gaze_estimator = GazeEstimator(device="cuda")
# passed to each session at add_session() time
```

**Recommendation**: Option B. L2CS-Net is stateless — one model instance shared across
all sessions. Saves ~100–200 MB VRAM and removes per-session model loading time.

### 5.7 MODIFIED: `core/proctor_coordinator.py`

Minor changes:
- If GazeEstimator is coordinator-level (Option B): instantiate it in `__init__` and
  inject into sessions via `add_session()`
- Optionally increase `_mp_workers` now that GIL is released:
  ```python
  # was capped at 8 for GIL-serial MP
  _mp_workers = min(max_sessions, 16)   # true parallel now — more workers = more throughput
  ```
- Rename `session.run_mediapipe` call if method is renamed (or keep name for zero diff)

### 5.8 MODIFIED: `config.py`

Updated threshold values for InsightFace degree-based pose (replace ratios with degrees):

```python
# Head pose — NOW IN DEGREES (was normalised ratio)
LOOK_AWAY_YAW   = 22     # degrees yaw to trigger "looking away"
LOOK_DOWN_PITCH = 14     # degrees pitch down
LOOK_UP_PITCH   = -10    # degrees pitch up

# Gaze — NOW IN RADIANS (was normalised ratio, L2CS-Net outputs radians)
GAZE_LEFT  = -0.15   # radians (~8-9°)
GAZE_RIGHT =  0.15   # radians (~8-9°)

# Partial face — unchanged (pixel dimensions, InsightFace bbox is also pixels)
MIN_FACE_WIDTH  = 110
MIN_FACE_HEIGHT = 120
```

**Calibration required**: The degree/radian values above are starting estimates based on
equivalence to the current ratio thresholds. Must be verified empirically by running
the debug overlay on a real webcam and checking that detection triggers at the right angles.

### 5.9 MODIFIED: `Dockerfile`

Add new dependencies layer:

```dockerfile
# After PyTorch layer, before requirements.txt:

# InsightFace system dep: libgomp1 already present (YOLO needs it too)
# ONNX Runtime GPU — install matching version for CUDA 12.4
RUN pip install --no-cache-dir \
        insightface \
        onnxruntime-gpu \
        l2cs

# Pre-download InsightFace buffalo_l model weights at build time
# (avoids first-run download delay in container)
RUN python3 -c "
import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l',
    allowed_modules=['detection','landmark_2d_106','landmark_3d_68'])
app.prepare(ctx_id=0, det_size=(320,320))
print('InsightFace buffalo_l downloaded')
"

# L2CS-Net weights — copy from local or download from HuggingFace
COPY models/L2CSNet_gaze360.pkl models/
```

**Note on `onnxruntime-gpu` vs `onnxruntime`**: Must install `onnxruntime-gpu` not
`onnxruntime` — they conflict, can't have both. InsightFace will auto-use CUDA EP when
`onnxruntime-gpu` is installed and `ctx_id=0` is passed.

**Note on `onnxruntime-gpu` version**: Pin to a version that matches CUDA 12.4.
As of mid-2025: `onnxruntime-gpu==1.18.0` supports CUDA 12.x.

### 5.10 MODIFIED: `requirements.txt`

```
# Remove:
mediapipe==0.10.11
protobuf==3.20.3

# Add:
insightface>=0.7.3
onnxruntime-gpu==1.18.0
l2cs>=0.1.1
```

Keep all other existing dependencies unchanged.

---

## 6. Architecture — What Changes, What Stays the Same

### Unchanged (no code changes needed)
- `core/proctor_coordinator.py` tick loop structure
- `core/proctor_session.py` `update()` method and all state machines
- `core/risk_engine.py`
- `core/alert_engine.py`
- `core/head_tracker.py`
- `core/liveness.py` (uses yaw/gaze values — just gets them from different source)
- `core/audio_monitor.py`
- `core/object_tracker.py`
- `core/metrics.py`
- `detectors/object_detector.py` (YOLO unchanged)
- `settings/scoring.py`
- `settings/alerts.py`
- `utils/` (all utils unchanged)
- `server.py` (all endpoints unchanged)
- `main.py`
- All frontend code

### New files
- `detectors/insightface_provider.py`
- `detectors/gaze_estimator.py`
- `models/L2CSNet_gaze360.pkl` (weights file — download or commit)

### Removed files
- `detectors/face_mesh_provider.py`
- `facemesh_worker.py` (top-level, ProcessPool leftover — clean it up)

### Modified files
- `detectors/head_pose_detector.py` (heavy)
- `detectors/lip_detector.py` (moderate)
- `detectors/__init__.py` (minor)
- `core/proctor_session.py` (moderate)
- `core/proctor_coordinator.py` (minor)
- `config.py` (thresholds only)
- `Dockerfile`
- `requirements.txt`

---

## 7. Expected Performance Gains

### Per-tick latency (g4dn.xlarge, 10 Hz target)

| Users | v1 tick (current) | v2 tick (estimated) | Headroom |
|-------|-------------------|---------------------|----------|
| 5 | ~90ms (MP=30ms) | ~70ms (IF=8ms) | 30ms to spare |
| 10 | ~150ms → drops to 7Hz | ~78ms → stays 10Hz | 22ms to spare |
| 15 | ~190ms → drops to 5Hz | ~82ms → stays 10Hz | 18ms to spare |
| 20 | not viable | ~90ms → ~9Hz | viable |

YOLO batch latency (~60ms) dominates in v2 — InsightFace is fully hidden behind it.

### Safe concurrent user target

- v1: **5 users per container** (MediaPipe bottleneck)
- v2: **12–15 users per container** (estimated — profile to confirm)

### VRAM impact

InsightFace buffalo_l (detection + landmark models only, no recognition):
- Approx 200–400 MB VRAM (ONNX Runtime allocates once, shared across sessions)
- L2CS-Net: ~50–100 MB VRAM
- Total new VRAM: ~300–500 MB on top of YOLO's ~1.5 GB
- T4 has 16 GB VRAM → ample headroom

---

## 8. Risks and Considerations

### 8.1 Landmark index verification (HIGH priority)

The InsightFace 106-point layout described in Section 4 is based on published model
documentation and community usage. **Verify all landmark indices empirically** by:
1. Running InsightFace on a test frame
2. Drawing each index as a numbered point on the frame
3. Confirming which indices correspond to eye corners, lip center, etc.

Incorrect indices will break EAR (blink) and MAR (lip) detection silently — no crash,
just wrong values.

### 8.2 Head pose units

InsightFace `face.pose` is in **degrees**. Current config uses **normalised ratios**.
The threshold values in Section 5.8 are estimates. Calibrate by:
1. Running debug overlay
2. Observing the yaw/pitch values at the visual boundary of "looking away"
3. Setting `LOOK_AWAY_YAW` to that value

### 8.3 L2CS-Net gaze calibration

Same as above — the radian thresholds for `GAZE_LEFT` / `GAZE_RIGHT` need empirical
calibration. The starting values (±0.15 rad ≈ ±8.6°) are estimates.

### 8.4 InsightFace detection confidence

InsightFace uses its own detection confidence internally. If it misses faces that
MediaPipe was catching (or vice versa), tune `det_size` and the detection threshold:
```python
app.prepare(ctx_id=0, det_size=(320, 320))
# For harder cases: det_size=(640, 640) — slower but higher recall
```

### 8.5 Thread safety of InsightFace per session

InsightFace `FaceAnalysis.get()` calls ONNX Runtime `session.run()`. ONNX Runtime
sessions are NOT thread-safe if the same session object is called from multiple threads.
Since each `ProctorSession` creates its own `InsightFaceProvider`, this is safe as long
as each session's `run_mediapipe()` is called from its assigned thread in the pool —
which is guaranteed by the current coordinator design.

### 8.6 InsightFace buffalo_l model download

The buffalo_l models are downloaded from InsightFace's CDN on first use. In Docker,
pre-download during build (see Dockerfile section). Otherwise first container start
will trigger a ~600 MB download.

### 8.7 ONNX Runtime version conflict

`onnxruntime` and `onnxruntime-gpu` are mutually exclusive packages. If any other
dependency pulls in `onnxruntime` (non-GPU), it will conflict. Check the full
dependency tree after installing. Pin `onnxruntime-gpu` explicitly and use
`pip install --no-deps insightface` if needed to avoid the conflict.

### 8.8 L2CS-Net weights licensing

L2CS-Net is MIT licensed. The pretrained weights (Gaze360 dataset) — verify the
dataset license before commercial use. For academic/internal proctoring use this is
fine.

### 8.9 Fallback behaviour when InsightFace misses a face

Current MediaPipe fallback: `landmarks=None` → all head/lip detections skip →
`face_detected=False` → liveness/fake_presence gates handled gracefully.

New fallback: `face=None` (InsightFace returned empty list) → same skip logic.
Ensure `head_detector.detect(frame, face=None)` and `lip_detector.process(frame, ts, face=None)`
return the same null-result tuples as v1 (`_NULL_HEAD`, `LipState(face_detected=False)`).

---

## 9. Implementation Order (Suggested)

1. **Set up the new repo** — clone, create venv, install new requirements
2. **Verify InsightFace works** — quick test script: load buffalo_l, run on webcam frame,
   print `face.pose`, draw all 106 landmarks with indices, print `face.bbox`
3. **Verify L2CS-Net works** — quick test script: pass face crop, print `(pitch, yaw)`
4. **Write `InsightFaceProvider`** — test in isolation
5. **Write `GazeEstimator`** — test in isolation
6. **Update `LipDetector`** — simplest change, no pose/gaze dependency
7. **Update `HeadPoseDetector`** — most complex, do last among detectors
8. **Update `ProctorSession`** — wire up new providers
9. **Update `ProctorCoordinator`** — inject GazeEstimator if coordinator-level
10. **Update `config.py`** — calibrate thresholds with debug overlay
11. **Run debug overlay** — visual sanity check on all features
12. **Load test** — benchmark at 5, 10, 15 users, compare tick latency to v1
13. **Update `Dockerfile`** — pre-download models, update deps layer
14. **Push and deploy**

---

## 10. Testing Checklist

Before declaring v2 stable, verify each detection visually with the debug overlay:

- [ ] Head yaw triggers "looking away" when turning head ~25° left/right
- [ ] Head pitch triggers "looking down" when looking at keyboard
- [ ] Head pitch triggers "looking up" when looking above screen
- [ ] Gaze triggers "looking side" when eyes shift (NOT just head turn)
- [ ] Gaze does NOT trigger when looking at camera directly
- [ ] Blink is detected (EAR drops, count increments)
- [ ] Blink is NOT detected on slow eye close (>400ms)
- [ ] Lip MAR increases when mouth opens
- [ ] Speaking detection triggers during continuous talking
- [ ] Yawn detection triggers on prolonged mouth open (>1.5s)
- [ ] Partial face triggers when face is partially off-screen
- [ ] Face hidden triggers when hand covers face (person still visible to YOLO)
- [ ] Face not detected → no crash, graceful `face_detected=False`
- [ ] All 5 concurrent users detect equally well from first tick

---

## 11. What Stays Exactly the Same

To be clear — the entire proctoring logic, scoring, and alerting pipeline is untouched:

- Risk engine thresholds and state machine
- Alert cooldowns and messages
- Tab switch detection and termination
- Object detection (YOLO) — phone, book, headphone, earbud
- Audio VAD (Silero)
- SSE streaming to admin
- All REST API endpoints
- Admin frontend
- Candidate frontend
- Docker deployment pattern (`--network host`, EC2, etc.)
- `SESSION_CONTEXT.md` patterns and scaling approach

Only the face landmark pipeline source changes. Everything downstream consumes the
same 12-element head pose tuple and `LipState` dataclass — zero changes required there.

---

*This document is the complete blueprint for v2. Begin implementation in a fresh clone.*
*Reference `CLAUDE.md` + `SESSION_CONTEXT.md` in the new repo for full system context.*
