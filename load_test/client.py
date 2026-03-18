"""
Single-candidate WebRTC load test client.

Uses aiortc to create a real RTCPeerConnection and stream a video file
to the backend exactly as a real browser would. The backend cannot
distinguish this from a real candidate.

Usage (standalone):
    python client.py --url https://<runpod-host>:8000 --video test.mp4
"""

import argparse
import asyncio
import logging
import ssl
import time

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

logger = logging.getLogger("load_test.client")


async def run_candidate(
    backend_url:  str,
    video_path:   str,
    candidate_id: str,
    duration_s:   int   = 300,
    results:      dict  = None,
    ssl_verify:   bool  = False,
) -> None:
    """
    Connect one simulated candidate to the backend and hold for duration_s.

    Args:
        backend_url:  Full URL of the backend, e.g. https://host:8000
        video_path:   Path to .mp4 file to stream (looped automatically)
        candidate_id: Unique label for this client (for logging/results)
        duration_s:   How long to hold the connection after it's established
        results:      Shared dict to write outcome into
        ssl_verify:   Set True to validate server TLS cert (False for RunPod)
    """
    t_start = time.perf_counter()
    pc = RTCPeerConnection()

    # MediaPlayer opens the file and provides VideoStreamTrack (+ AudioStreamTrack
    # if the file has audio).  loop=True means it restarts at EOF.
    player = MediaPlayer(video_path, loop=True)

    if player.video:
        pc.addTrack(player.video)
    if player.audio:
        pc.addTrack(player.audio)

    ssl_ctx = ssl.create_default_context()
    if not ssl_verify:
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE

    conn = aiohttp.TCPConnector(ssl=ssl_ctx)

    try:
        # ── SDP offer ────────────────────────────────────────────────────────
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        async with aiohttp.ClientSession(connector=conn) as http:
            async with http.post(
                f"{backend_url}/offer",
                json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    _record(results, candidate_id, "failed",
                            error=f"HTTP {resp.status}: {body[:200]}")
                    return

                data = await resp.json()

        # ── SDP answer ───────────────────────────────────────────────────────
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        )

        connect_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("[%s] connected in %.0f ms  pc_id=%s",
                     candidate_id, connect_ms, data.get("device_id", "?"))

        _record(results, candidate_id, "connected",
                connect_ms=round(connect_ms, 1),
                pc_id=data.get("device_id"),
                report_id=data.get("report_id"))

        # ── Hold for test duration ────────────────────────────────────────────
        await asyncio.sleep(duration_s)

        _record(results, candidate_id, "completed",
                connect_ms=round(connect_ms, 1),
                pc_id=data.get("device_id"))

    except asyncio.CancelledError:
        _record(results, candidate_id, "cancelled")
    except Exception as exc:
        logger.warning("[%s] error: %s", candidate_id, exc)
        _record(results, candidate_id, "error", error=str(exc))
    finally:
        await pc.close()


def _record(results: dict, candidate_id: str, status: str, **kwargs) -> None:
    if results is not None:
        entry = results.get(candidate_id, {})
        entry["status"] = status
        entry.update(kwargs)
        results[candidate_id] = entry


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--url",      required=True,  help="Backend URL, e.g. https://host:8000")
    ap.add_argument("--video",    required=True,  help="Path to .mp4 test video")
    ap.add_argument("--duration", type=int, default=60, help="Seconds to hold connection")
    args = ap.parse_args()

    results = {}
    asyncio.run(run_candidate(args.url, args.video, "test_candidate", args.duration, results))
    print(results)
