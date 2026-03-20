from __future__ import annotations

import logging
import time

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ── IoU / merge helpers ───────────────────────────────────────────────────────

def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


def merge_by_class(detections, classes, iou_threshold=0.5):
    final = []
    used  = set()
    grouped: dict = {}

    for i, d in enumerate(detections):
        if d["class"] in classes:
            grouped.setdefault(d["class"], []).append((i, d))
        else:
            final.append(d)

    for cls, items in grouped.items():
        clusters = []
        for idx, det in items:
            if idx in used:
                continue
            used.add(idx)
            cluster = [det]
            for jdx, other in items:
                if jdx in used:
                    continue
                if compute_iou(det["bbox"], other["bbox"]) >= iou_threshold:
                    cluster.append(other)
                    used.add(jdx)
            clusters.append(cluster)

        for cluster in clusters:
            best = max(
                cluster,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            )
            final.append(best)

    return final


# ── ObjectDetector ────────────────────────────────────────────────────────────

_ALLOWED = {"person", "cell_phone", "book", "headphone", "earbud"}


_MIN_VRAM_GB = 1.5   # overridden by ObjectDetector.__init__ from config


def _resolve_device(device: str, min_vram_gb: float = _MIN_VRAM_GB) -> tuple[str, str]:
    """
    Resolve the inference device, returning (device_str, reason).

    "auto" priority order:
      1. CUDA available + free VRAM >= min_vram_gb  → cuda   (NVIDIA / RunPod)
      2. CUDA unavailable, MPS available            → mps    (Apple Silicon M-series)
      3. Otherwise                                  → cpu

    Explicit choices:
      "cuda" → force NVIDIA GPU (warns if VRAM low, falls back to CPU if unavailable)
      "mps"  → force Apple GPU  (falls back to CPU if unavailable)
      "cpu"  → always CPU
    """
    if device == "cpu":
        return "cpu", "forced via --device cpu"

    # ── Explicit MPS ──────────────────────────────────────────────────────────
    if device == "mps":
        if torch.backends.mps.is_available():
            return "mps", "forced via --device mps (Apple Metal)"
        logger.warning("--device mps requested but MPS is not available — falling back to CPU")
        return "cpu", "MPS not available"

    cuda_ok = torch.cuda.is_available()

    # ── No CUDA: try MPS (Apple Silicon) before giving up on GPU ─────────────
    if not cuda_ok:
        if device == "cuda":
            logger.warning(
                "--device cuda requested but CUDA is not available — falling back to CPU"
            )
            return "cpu", "CUDA not available"
        # auto: try MPS next
        if torch.backends.mps.is_available():
            return "mps", "Apple MPS (Metal) — CUDA not available"
        return "cpu", "no GPU available (CUDA and MPS both absent)"

    # ── CUDA is present — check free VRAM before committing ──────────────────
    try:
        props      = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1e9
        used_vram  = torch.cuda.memory_allocated(0) / 1e9
        free_vram  = total_vram - used_vram

        if device == "auto" and free_vram < min_vram_gb:
            reason = f"free VRAM {free_vram:.1f} GB < {min_vram_gb:.1f} GB minimum"
            logger.warning("Auto device: falling back to CPU — %s", reason)
            return "cpu", reason

        if device == "cuda" and free_vram < min_vram_gb:
            logger.warning(
                "--device cuda: free VRAM is low (%.1f GB) — may cause OOM or system slowdown",
                free_vram,
            )

        return "cuda", f"GPU {props.name}  VRAM {free_vram:.1f}/{total_vram:.1f} GB free"

    except Exception as exc:
        logger.warning("VRAM check failed (%s) — falling back to CPU", exc)
        return "cpu", "VRAM check failed"


class ObjectDetector:
    """
    YOLO-based object detector with GPU/CPU auto-selection.

    GPU performance features:
      • FP16 (half-precision) inference  — ~2× throughput, ~½ VRAM  (CUDA only)
      • Warmup forward passes at startup — pre-compiles CUDA kernels,
        eliminates first-frame latency spike
      • Batch inference                  — one YOLO forward pass for N candidates

    device options (auto-detected or explicit):
      "auto"  — CUDA → MPS → CPU  (picks best available, default)
      "cuda"  — force NVIDIA GPU  (RunPod / Linux with NVIDIA)
      "mps"   — force Apple GPU   (MacBook M-series)
      "cpu"   — force CPU
    """

    def __init__(
        self,
        model_path:    str   = "finalBestV5.pt",
        device:        str   = "cpu",
        default_conf:  float = 0.50,
        person_conf:   float = 0.30,
        phone_conf:    float = 0.65,
        book_conf:     float = 0.70,
        audio_conf:    float = 0.41,
        half:          bool  = False,   # disabled by default — opt-in for confirmed GPU setups
        warmup_frames: int   = 0,       # disabled by default — avoids blocking event loop at startup
        min_vram_gb:   float = 1.5,     # auto-mode: fall back to CPU below this free VRAM
        imgsz:         int   = 640,     # inference resolution; set at load time, NOT at call time
    ):
        # Resolve device with VRAM safety check
        self.device, _reason = _resolve_device(device, min_vram_gb)
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Set inference resolution at load time — safe for batch mode.
        # NEVER pass imgsz at inference time: it breaks orig_shape tracking in
        # Ultralytics batch mode and shifts all bboxes to wrong pixel positions.
        if imgsz != 640:
            self.model.overrides['imgsz'] = imgsz
            logger.info("ObjectDetector: inference imgsz set to %d (load-time override)", imgsz)

        # FP16: CUDA only — CPU half-precision is slower; MPS float16 support
        # is still maturing across PyTorch versions so we keep MPS on FP32.
        self._half = half and self.device == "cuda"
        if self._half:
            self.model.half()

        logger.info(
            "ObjectDetector ready  device=%s  half=%s  reason=\"%s\"",
            self.device, self._half, _reason,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(
                "GPU: %s  VRAM total=%.1f GB  compute=%d.%d",
                props.name, props.total_memory / 1e9, props.major, props.minor,
            )
        elif self.device == "mps":
            logger.info("GPU: Apple MPS (Metal Performance Shaders)")

        self.default_conf     = default_conf
        self.person_conf      = person_conf
        self.class_thresholds = {
            "cell_phone": phone_conf,
            "book"      : book_conf,
            "headphone" : audio_conf,
            "earbud"    : audio_conf,
        }

        # ONNX models exported with fixed batch=1 must fall back to sequential calls
        self._supports_batch: bool = not str(model_path).lower().endswith(".onnx")

        # Timing stats for /metrics endpoint
        self._last_batch_ms: float = 0.0
        self._total_batches: int   = 0
        self._total_frames:  int   = 0

        # Warmup: schedule in a daemon thread so it never blocks the asyncio event loop.
        # The first N real frames will trigger kernel compilation instead — on GPU the
        # latency difference is only 1–2 frames.  0 = disabled (default).
        if warmup_frames > 0:
            self._schedule_warmup(warmup_frames)

    def _warmup(self, n: int) -> None:
        """Run N dummy inference passes. Called in a background thread."""
        logger.info("ObjectDetector: warmup starting (%d pass(es))…", n)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(n):
            try:
                self.model([dummy], verbose=False, device=self.device, imgsz=640)
            except Exception as exc:
                logger.warning("ObjectDetector: warmup pass failed — %s", exc)
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ObjectDetector: warmup done  %.0f ms total  (%.0f ms/pass)",
            elapsed_ms, elapsed_ms / n,
        )

    def _schedule_warmup(self, n: int) -> None:
        """Fire warmup in a daemon thread — never blocks the caller."""
        import threading
        threading.Thread(
            target=self._warmup, args=(n,),
            daemon=True, name="yolo-warmup",
        ).start()

    @property
    def device_info(self) -> dict:
        """Return human-readable device information for /system/report."""
        info: dict = {
            "device"          : self.device,
            "half_precision"  : self._half,
            "cuda_available"  : torch.cuda.is_available(),
            "mps_available"   : torch.backends.mps.is_available(),
            "batch_supported" : self._supports_batch,
            "total_batches"   : self._total_batches,
            "total_frames"    : self._total_frames,
            "last_batch_ms"   : round(self._last_batch_ms, 2),
        }
        if self.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name"           : props.name,
                "gpu_vram_gb"        : round(props.total_memory / 1e9, 1),
                "gpu_compute"        : f"{props.major}.{props.minor}",
                "gpu_mem_alloc_mb"   : round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
            })
        elif self.device == "mps":
            info["gpu_name"] = "Apple MPS (Metal)"
            try:
                info["mps_alloc_mb"]  = round(torch.mps.current_allocated_memory() / 1024 / 1024, 1)
                info["mps_driver_mb"] = round(torch.mps.driver_allocated_memory()   / 1024 / 1024, 1)
            except Exception:
                pass
        return info

    def _parse_result(self, r) -> list[dict]:
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name   = self.model.names[cls_id]
            conf   = float(box.conf[0])

            if name not in _ALLOWED:
                continue

            threshold = self.class_thresholds.get(name, self.default_conf)
            if conf < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class"     : name,
                "confidence": conf,
                "bbox"      : (x1, y1, x2, y2),
            })
        return detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO on a single frame.
        Do NOT pass imgsz at call time — Ultralytics uses the model's native
        640px and guarantees that box.xyxy coords are in the original frame's
        pixel space.  Passing imgsz at runtime breaks the per-frame orig_shape
        tracking in batch mode and shifts all bboxes to wrong positions.
        """
        results = self.model(frame, verbose=False, device=self.device)
        return self._parse_result(results[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """
        Run YOLO over a batch of frames in a single forward pass.
        Do NOT pass imgsz at call time — see detect() for the reason.
        Output bbox coords are always in each frame's original pixel space.
        """
        if not frames:
            return []

        t0 = time.perf_counter()

        if self._supports_batch:
            all_results = self.model(frames, verbose=False, device=self.device)
            out = [self._parse_result(r) for r in all_results]
        else:
            out = [self.detect(f) for f in frames]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._last_batch_ms  = elapsed_ms
        self._total_batches += 1
        self._total_frames  += len(frames)

        # Push to metrics singleton (imported lazily to avoid circular import)
        try:
            from core.metrics import metrics as _m
            _m.record_yolo_latency(elapsed_ms)
        except Exception:
            pass

        return out
