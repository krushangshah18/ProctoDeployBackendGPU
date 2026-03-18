"""
local_runner.py — development/testing mode.

Runs the full proctoring pipeline against the local webcam + microphone
without WebRTC. Displays the raw camera feed in an OpenCV window with
alert/risk state printed to console. Press 'q' to quit.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import config
from core.proctor_coordinator import ProctorCoordinator
from core.proctor_session import ProctorSession


def get_config() -> dict:
    return {k: getattr(config, k) for k in dir(config) if k.isupper()}


async def main():
    cfg = get_config()
    cfg.setdefault("YOLO_MODEL_PATH", "finalBestV5.pt")

    logger.info("Starting ProctorCoordinator...")
    coordinator = ProctorCoordinator(
        model_path   = cfg["YOLO_MODEL_PATH"],
        tick_rate    = cfg.get("TICK_RATE", 10),
        max_sessions = cfg.get("MAX_SESSIONS", 40),
        default_conf = cfg.get("YOLO_DEFAULT_CONF", 0.50),
        person_conf  = cfg.get("YOLO_PERSON_CONF",  0.30),
        phone_conf   = cfg.get("YOLO_PHONE_CONF",   0.65),
        book_conf    = cfg.get("YOLO_BOOK_CONF",    0.70),
        audio_conf   = cfg.get("YOLO_AUDIO_CONF",   0.41),
    )
    await coordinator.start()

    session_id  = "Candidate_1"
    ts          = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path("records") / f"{ts}_{session_id}"

    session = ProctorSession(
        session_id       = session_id,
        session_dir      = session_dir,
        config           = cfg,
        use_webrtc_audio = False,   # use local microphone via pyaudio
    )
    coordinator.add_session(session_id, session)

    logger.info("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        await coordinator.stop()
        return

    for _ in range(5):   # warm-up
        cap.read()
        time.sleep(0.1)

    logger.info("Local AI Proctor running — press 'q' to quit.")
    cv2.namedWindow("Local AI Proctor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Local AI Proctor", 800, 600)

    last_frame_time = time.time()
    last_log        = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break

            now = time.time()
            fps = 1.0 / (now - last_frame_time) if now > last_frame_time else 15.0
            last_frame_time = now

            session.latest_frame = frame
            session.observed_fps = fps

            # Display raw frame (no drawing in production mode)
            cv2.imshow("Local AI Proctor", frame)

            # Print risk/alert state to console every 2 s
            if now - last_log >= 2.0:
                last_log = now
                risk_info = session.risk.get_display()
                alerts    = session._alert_manager.get_active_alerts()
                warnings  = session._alert_manager.get_active_warnings()
                logger.info(
                    "Risk: %.0f (%s) | alerts=%d warnings=%d",
                    risk_info["score"], risk_info["state"],
                    len(alerts), len(warnings),
                )
                for a in alerts:
                    logger.warning("ALERT: %s", a)

            await asyncio.sleep(0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        logger.info("Shutting down...")
        coordinator.remove_session(session_id)
        await coordinator.stop()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done. Report saved in records/ folder.")


if __name__ == "__main__":
    asyncio.run(main())
