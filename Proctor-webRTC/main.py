"""
Production entry point.

Run:
    python main.py                         # auto device, HTTP
    python main.py --device cpu            # force CPU inference
    python main.py --device cuda           # force GPU inference
    python main.py --device auto --half    # GPU FP16 (if VRAM ok)
    python main.py --warmup 3              # pre-compile CUDA kernels
    python main.py --port 9000             # custom port
    LOG_LEVEL=DEBUG python main.py         # verbose logging
    uvicorn server:app --workers 1         # direct uvicorn invocation
"""
import os
from pathlib import Path

# ── Must be first — sets up root logger before any other import ───────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils.logging_config import setup_logging
setup_logging()

import argparse
import logging
import uvicorn

logger = logging.getLogger("main")

KEY_FILE  = Path.home() / "key.pem"
CERT_FILE = Path.home() / "cert.pem"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Proctor Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Device selection:
  auto  — use CUDA if free VRAM >= min_vram, otherwise fall back to CPU (default)
  cuda  — force GPU (warns if VRAM is low)
  cpu   — force CPU

Examples:
  python main.py
  python main.py --device cpu
  python main.py --device cuda --half --warmup 3
  python main.py --port 9000
""",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device: auto (CUDA→MPS→CPU), cuda, mps (Apple), or cpu  (default: auto)",
    )
    parser.add_argument(
        "--half", action="store_true", default=False,
        help="Enable FP16 half-precision on CUDA (~2× throughput, ~½ VRAM). Ignored on CPU.",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, metavar="N",
        help="Number of YOLO warmup passes at startup to pre-compile CUDA kernels (default: 0 = off).",
    )
    parser.add_argument(
        "--port", type=int, default=8000, metavar="PORT",
        help="Port to listen on (default: 8000).",
    )
    args = parser.parse_args()

    # Pass CLI choices to server.py via environment variables.
    # server.py reads these in lifespan() before constructing the coordinator.
    os.environ["PROCTOR_DEVICE"] = args.device
    os.environ["PROCTOR_HALF"]   = "1" if args.half else "0"
    os.environ["PROCTOR_WARMUP"] = str(args.warmup)

    use_ssl = KEY_FILE.exists() and CERT_FILE.exists()
    scheme  = "https" if use_ssl else "http"

    logger.info(
        "Starting AI Proctor server  %s://0.0.0.0:%d  ssl=%s  device=%s  half=%s  warmup=%d",
        scheme, args.port, use_ssl, args.device, args.half, args.warmup,
    )

    uvicorn.run(
        "server:app",
        host              = "0.0.0.0",
        port              = args.port,
        # Single worker — the model and WebRTC peer connections are
        # process-local; multi-worker would create duplicate coordinators.
        workers           = 1,
        # Use uvicorn's own access log so requests appear in stdout
        access_log        = True,
        # Forward uvicorn/asyncio logs through our structured handlers
        log_config        = None,
        # Generous timeout — WebRTC ICE negotiation can take several seconds
        timeout_keep_alive= 30,
        ssl_keyfile       = str(KEY_FILE)  if use_ssl else None,
        ssl_certfile      = str(CERT_FILE) if use_ssl else None,
    )
