---
name: Feedback and working preferences
description: How the user wants code changes approached and what to avoid
type: feedback
---

**Remove dead code completely, don't leave it commented out.**
Why: Production readiness audit — user explicitly asked to strip all unused code paths (CPU fallbacks, MODEL_ENABLED flag, old dev files).
How to apply: When making changes, delete unused code rather than commenting it or guarding with a flag.

**Never use module-level import bindings as live config for per-session objects.**
Why: HeadPoseDetector was reading from `from config import MIN_FACE_WIDTH` which is fixed at import time — runtime_settings changes didn't apply to new sessions.
How to apply: Always pass runtime-configurable values as explicit constructor params, reading from session_cfg.

**Fail fast on missing GPU, don't fall back to CPU silently.**
Why: Production is GPU-only. Silent CPU fallback would just run degraded without anyone noticing.
How to apply: Use `_ensure_cuda()` pattern — raise RuntimeError if CUDA unavailable.

**Keep docker run commands without `--device` arg.**
Why: `--device` CLI arg was removed from main.py. Old commands with `--device auto` caused container crash-loops.
How to apply: Docker run CMD is `python main.py --half --warmup 3 --port <PORT>`.

**Don't run warmup or heavy init synchronously in lifespan/startup.**
Why: Blocked the asyncio event loop, preventing server from accepting connections.
How to apply: Use daemon background thread (`threading.Thread(daemon=True)`) for warmup and similar one-time startup tasks.
