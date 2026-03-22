"""
Production entry point — GPU (CUDA) required.

Run:
    python main.py                  # default: FP16, 3 warmup passes, port 8000
    python main.py --warmup 3       # pre-compile CUDA kernels at startup
    python main.py --half           # enable FP16 (already on by default in config)
    python main.py --port 9000      # custom port
    LOG_LEVEL=DEBUG python main.py  # verbose logging
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

# uvloop replaces asyncio's default event loop with libuv — significantly lower
# per-coroutine overhead.  Optional dep: falls back silently if not installed.
try:
    import uvloop
    uvloop.install()
    logging.getLogger("main").info("uvloop installed — using libuv event loop")
except ImportError:
    pass

logger = logging.getLogger("main")

KEY_FILE  = Path.home() / "key.pem"
CERT_FILE = Path.home() / "cert.pem"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Proctor Server (GPU)")
    parser.add_argument(
        "--half", action="store_true", default=False,
        help="Enable FP16 half-precision (~2× throughput, ~½ VRAM). Default from config.py.",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, metavar="N",
        help="Number of YOLO warmup passes at startup to pre-compile CUDA kernels (default: 0 = use config).",
    )
    parser.add_argument(
        "--port", type=int, default=8000, metavar="PORT",
        help="Port to listen on (default: 8000).",
    )
    args = parser.parse_args()

    # Pass CLI choices to server.py via environment variables.
    os.environ["PROCTOR_HALF"]   = "1" if args.half else "0"
    os.environ["PROCTOR_WARMUP"] = str(args.warmup)

    use_ssl = KEY_FILE.exists() and CERT_FILE.exists()
    scheme  = "https" if use_ssl else "http"

    logger.info(
        "Starting AI Proctor server  %s://0.0.0.0:%d  ssl=%s  half=%s  warmup=%d",
        scheme, args.port, use_ssl, args.half, args.warmup,
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
