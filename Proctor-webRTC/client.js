// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusEl       = document.getElementById("status");
const btnStart       = document.getElementById("btn-start");
const btnEnd         = document.getElementById("btn-end");
const indicator      = document.getElementById("indicator");
const indicatorLabel = document.getElementById("indicator-label");

// ── State ─────────────────────────────────────────────────────────────────────
let pc           = null;
let stream       = null;
let deviceId     = null;
let statsTimer   = null;
let srvTimer     = null;
let alertsTimer  = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(msg) {
  console.log(msg);
  if (statusEl) statusEl.textContent = msg;
}

function setActive(active) {
  btnStart.disabled = active;
  btnEnd.disabled   = !active;
  indicator.className = active ? "active" : "";
  indicatorLabel.textContent = active ? "Live" : "Idle";
  indicatorLabel.style.color = active ? "#e74c3c" : "#666";
}

function colorVal(el, value, warnThresh, badThresh, higherIsBad = true) {
  if (value == null || el == null) return;
  const n = parseFloat(value);
  el.className = "stat-value";
  if (higherIsBad) {
    if (n >= badThresh) el.classList.add("bad");
    else if (n >= warnThresh) el.classList.add("warn");
  } else {
    if (n <= badThresh) el.classList.add("bad");
    else if (n <= warnThresh) el.classList.add("warn");
  }
}

function resetStats() {
  const ids = ["s-fps","s-res","s-vbr","s-lost","s-rtt",
               "srv-fps","srv-jitter","srv-interval","srv-frames","srv-res",
               "s-abr","srv-arate","srv-asr","srv-ach"];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) { el.textContent = "—"; el.className = "stat-value"; }
  });
  window._videoPrev = null;
  window._audioPrev = null;
}

// ── Server-side stats poll ─────────────────────────────────────────────────────
async function pollServerStats() {
  if (!deviceId) return;
  try {
    const res = await fetch("/stats");
    const all = await res.json();
    const s   = all[deviceId];
    if (!s) return;

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el && val != null) el.textContent = val;
    };

    set("srv-fps",      s.fps != null ? `${s.fps} fps` : null);
    colorVal(document.getElementById("srv-fps"), s.fps, 20, 10, false);

    set("srv-jitter",   s.jitter_ms != null ? `${s.jitter_ms} ms` : null);
    colorVal(document.getElementById("srv-jitter"), s.jitter_ms, 10, 30);

    set("srv-interval", s.mean_interval_ms != null ? `${s.mean_interval_ms} ms` : null);
    set("srv-frames",   s.total_frames);
    set("srv-res",      s.resolution);
    set("srv-arate",    s.audio_packet_rate != null ? `${s.audio_packet_rate}/s` : null);
    set("srv-asr",      s.audio_sample_rate != null ? `${s.audio_sample_rate} Hz` : null);
    set("srv-ach",      s.audio_channels);
  } catch (_) {}
}

// ── Client-side WebRTC stats poll ─────────────────────────────────────────────
async function pollClientStats() {
  if (!pc) return;
  const stats = await pc.getStats();

  stats.forEach((report) => {
    if (report.type === "outbound-rtp" && report.kind === "video") {
      const fpsEl = document.getElementById("s-fps");
      fpsEl.textContent = `${report.framesPerSecond ?? "—"} fps`;
      colorVal(fpsEl, report.framesPerSecond, 20, 10, false);

      document.getElementById("s-res").textContent =
        report.frameWidth ? `${report.frameWidth}x${report.frameHeight}` : "—";

      if (report.bytesSent != null) {
        window._videoPrev = window._videoPrev || {};
        const prev = window._videoPrev;
        if (prev.bytes != null && prev.ts != null) {
          const dt  = (report.timestamp - prev.ts) / 1000;
          const bps = ((report.bytesSent - prev.bytes) * 8) / dt;
          document.getElementById("s-vbr").textContent = `${(bps / 1000).toFixed(0)} kbps`;
        }
        prev.bytes = report.bytesSent;
        prev.ts    = report.timestamp;
      }
    }

    if (report.type === "outbound-rtp" && report.kind === "audio") {
      window._audioPrev = window._audioPrev || {};
      const prev = window._audioPrev;
      if (prev.bytes != null && prev.ts != null && report.bytesSent != null) {
        const dt  = (report.timestamp - prev.ts) / 1000;
        const bps = ((report.bytesSent - prev.bytes) * 8) / dt;
        document.getElementById("s-abr").textContent = `${(bps / 1000).toFixed(1)} kbps`;
      }
      if (report.bytesSent != null) {
        prev.bytes = report.bytesSent;
        prev.ts    = report.timestamp;
      }
    }

    if (report.type === "remote-inbound-rtp" && report.kind === "video") {
      const lostEl = document.getElementById("s-lost");
      const lost   = report.packetsLost ?? 0;
      lostEl.textContent = lost;
      colorVal(lostEl, lost, 5, 20);

      if (report.roundTripTime != null) {
        const rtt   = (report.roundTripTime * 1000).toFixed(1);
        const rttEl = document.getElementById("s-rtt");
        rttEl.textContent = `${rtt} ms`;
        colorVal(rttEl, rtt, 50, 150);
      }
    }
  });
}

// ── Alert / Warning banners ───────────────────────────────────────────────────
// Persistent map: key → { el, expiry } so banners stay visible for a fixed
// duration and don't flicker on every poll cycle.
const _activeBanners = new Map();
const ALERT_TTL_MS   = 6000;   // keep alert banners for 6 s
const WARN_TTL_MS    = 4000;   // keep warning banners for 4 s

async function pollAlerts() {
  if (!deviceId) return;
  const container = document.getElementById("alert-container");
  if (!container) return;
  const now = Date.now();

  try {
    const res  = await fetch(`/alerts/${deviceId}`);
    if (!res.ok) return;
    const data = await res.json();

    // Add / refresh alerts
    (data.alerts || []).forEach((msg) => {
      const key = "alert:" + msg;
      if (_activeBanners.has(key)) {
        _activeBanners.get(key).expiry = now + ALERT_TTL_MS;
      } else {
        const el = document.createElement("div");
        el.className = "banner banner-alert";
        el.innerHTML = `<span class="banner-icon">&#9888;</span>${msg}`;
        container.appendChild(el);
        _activeBanners.set(key, { el, expiry: now + ALERT_TTL_MS });
      }
    });

    // Add / refresh warnings
    (data.warnings || []).forEach((msg) => {
      const key = "warn:" + msg;
      if (_activeBanners.has(key)) {
        _activeBanners.get(key).expiry = now + WARN_TTL_MS;
      } else {
        const el = document.createElement("div");
        el.className = "banner banner-warn";
        el.innerHTML = `<span class="banner-icon">&#9888;</span>${msg}`;
        container.appendChild(el);
        _activeBanners.set(key, { el, expiry: now + WARN_TTL_MS });
      }
    });
  } catch (_) {}

  // Remove banners whose TTL has elapsed
  for (const [key, entry] of _activeBanners) {
    if (now > entry.expiry) {
      entry.el.remove();
      _activeBanners.delete(key);
    }
  }
}

function clearAlerts() {
  _activeBanners.clear();
  const container = document.getElementById("alert-container");
  if (container) container.innerHTML = "";
}

// ── Capacity check ────────────────────────────────────────────────────────────
async function checkCapacity() {
  const res  = await fetch("/capacity");
  const data = await res.json();
  if (!data.available) {
    throw new Error(`All ${data.max} candidate slots are full. Try again later.`);
  }
}

// ── Start Session ─────────────────────────────────────────────────────────────
async function startSession() {
  btnStart.disabled = true;
  setStatus("Checking server capacity...");

  await checkCapacity();

  setStatus("Requesting camera/mic access...");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus(
      "Error: Camera access requires a secure context. " +
      "Open this page at http://localhost:8080 (not via an IP address), " +
      "or serve over HTTPS."
    );
    btnStart.disabled = false;
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch (err) {
    console.warn("No audio, trying video-only:", err.message);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setStatus("Warning: No microphone — streaming video only.");
    } catch (err2) {
      setStatus(`Error: ${err2.message}`);
      btnStart.disabled = false;
      return;
    }
  }

  document.getElementById("video").srcObject = stream;

  pc = new RTCPeerConnection();
  pc.onconnectionstatechange = () => {
    setStatus(`Connection: ${pc.connectionState}`);
    if (pc.connectionState === "connected") setActive(true);
    if (["failed", "closed", "disconnected"].includes(pc.connectionState)) setActive(false);
  };

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  setStatus("Gathering ICE candidates...");
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") resolve();
    else pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") resolve();
    });
  });

  setStatus("Connecting to server...");
  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });

  const answer = await response.json();

  if (response.status === 503) {
    stream.getTracks().forEach((t) => t.stop());
    pc.close();
    pc = null; stream = null; deviceId = null;
    setActive(false);
    btnStart.disabled = false;
    setStatus(`Cannot connect: ${answer.error}`);
    return;
  }

  deviceId = answer.device_id;
  await pc.setRemoteDescription(answer);

  if (answer.device_label) {
    document.querySelector("h2").textContent = `AI Proctor — ${answer.device_label}`;
  }

  statsTimer  = setInterval(pollClientStats, 1000);
  srvTimer    = setInterval(pollServerStats, 1000);
  alertsTimer = setInterval(pollAlerts,      500);
}

// ── End Session ───────────────────────────────────────────────────────────────
async function endSession() {
  btnEnd.disabled = true;
  setStatus("Stopping session...");

  clearInterval(statsTimer);
  clearInterval(srvTimer);
  clearInterval(alertsTimer);
  statsTimer  = null;
  srvTimer    = null;
  alertsTimer = null;
  clearAlerts();

  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (pc)     { pc.close(); pc = null; }

  document.getElementById("video").srcObject = null;
  deviceId = null;
  resetStats();
  setActive(false);
  setStatus("Session ended.");
}

// ── Button bindings ───────────────────────────────────────────────────────────
btnStart.addEventListener("click", () => startSession().catch((err) => {
  setStatus(`Error: ${err.message}`);
  setActive(false);
  btnStart.disabled = false;
}));

btnEnd.addEventListener("click", endSession);
