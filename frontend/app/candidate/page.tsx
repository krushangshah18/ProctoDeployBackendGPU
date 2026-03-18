"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { SSEEvent, RiskInfo } from "@/lib/types";

const RISK_COLORS: Record<string, string> = {
  NORMAL      : "#22c55e",
  WARNING     : "#f59e0b",
  HIGH_RISK   : "#ef4444",
  ADMIN_REVIEW: "#dc2626",
  TERMINATED  : "#7f1d1d",
};

interface Banner {
  id: string;
  type: "alert" | "warning";
  message: string;
  expiresAt: number;
}

export default function CandidatePage() {
  const videoRef          = useRef<HTMLVideoElement>(null);
  const pcRef             = useRef<RTCPeerConnection | null>(null);
  const streamRef         = useRef<MediaStream | null>(null);
  const sseRef            = useRef<EventSource | null>(null);
  // Refs so event handlers always have the latest values without stale closures
  const pcIdRef              = useRef<string | null>(null);
  const statusRef            = useRef<string>("idle");
  const connectedAtRef       = useRef<number>(0);    // wall-clock ms when exam connected
  const tabSwitchCountRef    = useRef<number>(0);    // authoritative count (no stale closure risk)
  const terminationShownRef  = useRef<boolean>(false); // show termination banner only once
  const lastCopyWarnRef      = useRef<number>(0);    // cooldown for copy-paste warning banners

  const [status, setStatus]     = useState<"idle" | "connecting" | "connected" | "ended">("idle");
  const [error, setError]       = useState("");
  const [pcId, setPcId]         = useState<string | null>(null);
  const [deviceLabel, setLabel] = useState("");
  const [risk, setRisk]         = useState<RiskInfo | null>(null);
  const [banners, setBanners]   = useState<Banner[]>([]);
  const [tabSwitchCount, setTabSwitchCount] = useState(0);

  // ── Banner management ───────────────────────────────────────────────────
  const addBanner = useCallback((type: "alert" | "warning", message: string) => {
    const id        = `${Date.now()}-${Math.random()}`;
    const expiresAt = Date.now() + (type === "alert" ? 8000 : 5000);
    setBanners(prev => [...prev.slice(-4), { id, type, message, expiresAt }]);
  }, []);

  // Expire banners
  useEffect(() => {
    const t = setInterval(() => {
      setBanners(prev => prev.filter(b => b.expiresAt > Date.now()));
    }, 500);
    return () => clearInterval(t);
  }, []);

  // Keep refs in sync so event handlers always see the latest values
  useEffect(() => { pcIdRef.current   = pcId;   }, [pcId]);
  useEffect(() => { statusRef.current = status; }, [status]);
  // Record exact moment the exam connected (used for 2-second grace period below)
  useEffect(() => {
    if (status === "connected") connectedAtRef.current = Date.now();
  }, [status]);

  // ── Shared handler: report a focus-loss violation to the backend ─────────
  // Used by both tab-switch and window-blur handlers to avoid duplication.
  const reportFocusViolation = useCallback((reason: "tab_switch" | "window_blur") => {
    if (statusRef.current !== "connected" || !pcIdRef.current) return;
    if (Date.now() - connectedAtRef.current < 2000) return;

    tabSwitchCountRef.current += 1;
    const count = tabSwitchCountRef.current;
    setTabSwitchCount(count);

    api.tabSwitch(pcIdRef.current).then(result => {
      setRisk(result.risk);

      if (result.risk.terminated) {
        if (!terminationShownRef.current) {
          terminationShownRef.current = true;
          addBanner("alert", `EXAM TERMINATED — left exam ${count} time${count !== 1 ? "s" : ""}`);
        }
      } else {
        const label = reason === "window_blur" ? "Focus lost (split screen)" : "Tab switch detected";
        addBanner("alert", `${label} — violation ${count}/3 (exam terminates at 3)`);
      }
    }).catch(() => {});
  }, [addBanner]);

  // ── Tab switch detection (visibilitychange) ──────────────────────────────
  // WHY we don't use SSE here: when the tab is hidden the browser pauses/throttles
  // the EventSource connection, so any SSE event the server sends while the tab is
  // hidden is lost before the candidate can see it. Instead we:
  //   • call POST /tab_switch immediately (fetch works fine in hidden tabs)
  //   • read the risk update directly from the response JSON
  //   • show the appropriate banner ourselves without waiting for SSE
  //   • when the tab becomes visible again we re-fetch risk to catch any other
  //     missed SSE events (e.g. termination via no_person / multiple_people)
  useEffect(() => {
    const handleVisibility = () => {
      if (document.hidden) {
        reportFocusViolation("tab_switch");
      } else {
        // Tab became visible — re-sync risk state to catch any SSE events that
        // were lost while hidden (e.g. termination from no_person / multiple_people).
        if (statusRef.current === "connected" && pcIdRef.current) {
          api.risk(pcIdRef.current).then(r => {
            setRisk(r);
            if (r.terminated && !terminationShownRef.current) {
              terminationShownRef.current = true;
              addBanner("alert", "EXAM TERMINATED — check the risk panel");
            }
          }).catch(() => {});
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [addBanner, reportFocusViolation]);

  // ── Window blur detection (split screen / alt-tab to another app) ────────
  // visibilitychange only fires when the tab becomes hidden (tab switch,
  // window minimise). In split-screen the tab stays VISIBLE but the browser
  // window loses keyboard/mouse focus — only window.blur fires in that case.
  //
  // Problem: window.blur ALSO fires just before visibilitychange (tab switch).
  // Solution: delay 150 ms and skip if the tab is now hidden (already counted).
  useEffect(() => {
    const handleBlur = () => {
      setTimeout(() => {
        if (document.hidden) return;    // tab-switch — already handled above
        reportFocusViolation("window_blur");
      }, 150);
    };

    const handleFocus = () => {
      // Window regained focus — sync risk in case something happened while away
      if (statusRef.current === "connected" && pcIdRef.current) {
        api.risk(pcIdRef.current).then(r => {
          setRisk(r);
          if (r.terminated && !terminationShownRef.current) {
            terminationShownRef.current = true;
            addBanner("alert", "EXAM TERMINATED — check the risk panel");
          }
        }).catch(() => {});
      }
    };

    window.addEventListener("blur",  handleBlur);
    window.addEventListener("focus", handleFocus);
    return () => {
      window.removeEventListener("blur",  handleBlur);
      window.removeEventListener("focus", handleFocus);
    };
  }, [addBanner, reportFocusViolation]);

  // ── Copy / paste / cut prevention ───────────────────────────────────────
  // Active only while the exam is live. Shows a warning banner on attempt
  // with a 4-second cooldown to avoid spamming.
  useEffect(() => {
    if (status !== "connected") return;

    const warn = () => {
      const now = Date.now();
      if (now - lastCopyWarnRef.current < 4000) return;
      lastCopyWarnRef.current = now;
      addBanner("warning", "Copy / paste is disabled during the exam");
    };

    const blockClipboard = (e: ClipboardEvent) => { e.preventDefault(); warn(); };

    const blockKeys = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && ["c", "v", "x"].includes(e.key.toLowerCase())) {
        e.preventDefault();
        warn();
      }
    };

    document.addEventListener("copy",  blockClipboard);
    document.addEventListener("cut",   blockClipboard);
    document.addEventListener("paste", blockClipboard);
    document.addEventListener("keydown", blockKeys);

    return () => {
      document.removeEventListener("copy",  blockClipboard);
      document.removeEventListener("cut",   blockClipboard);
      document.removeEventListener("paste", blockClipboard);
      document.removeEventListener("keydown", blockKeys);
    };
  }, [status, addBanner]);

  // ── SSE connection ──────────────────────────────────────────────────────
  const connectSSE = useCallback((id: string) => {
    if (sseRef.current) sseRef.current.close();

    const es = new EventSource(api.streamUrl(id));
    sseRef.current = es;

    es.onmessage = (e) => {
      try {
        const event: SSEEvent = JSON.parse(e.data);
        if (event.risk) setRisk(event.risk);

        if (event.type === "alert") {
          addBanner("alert", event.message || event.key || "Alert");
        } else if (event.type === "warning") {
          addBanner("warning", event.message || "Warning");
        } else if (event.type === "session_end") {
          setStatus("ended");
          es.close();
        }
      } catch { /* ignore parse errors */ }
    };

    es.onerror = () => {
      // EventSource auto-reconnects; only log
    };
  }, [addBanner]);

  // ── WebRTC ─────────────────────────────────────────────────────────────
  const startSession = async () => {
    setError("");
    setStatus("connecting");
    setTabSwitchCount(0);
    tabSwitchCountRef.current   = 0;
    terminationShownRef.current = false;

    try {
      // 1. Get camera + mic
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .catch(() => navigator.mediaDevices.getUserMedia({ video: true, audio: false }));

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // 2. Create peer connection
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      stream.getTracks().forEach(t => pc.addTrack(t, stream));

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "connected") setStatus("connected");
        if (["failed", "closed", "disconnected"].includes(pc.connectionState)) {
          setStatus("ended");
        }
      };

      // 3. Create offer & wait for ICE
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await new Promise<void>(resolve => {
        if (pc.iceGatheringState === "complete") { resolve(); return; }
        pc.addEventListener("icegatheringstatechange", () => {
          if (pc.iceGatheringState === "complete") resolve();
        });
      });

      // 4. Send to server
      const answer = await api.offer(pc.localDescription!.sdp, pc.localDescription!.type);

      if (answer.error) {
        throw new Error(answer.error);
      }

      setPcId(answer.device_id);
      setLabel(answer.device_label);
      await pc.setRemoteDescription(answer);

      // 5. Connect SSE for real-time alerts
      connectSSE(answer.device_id);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setStatus("idle");
      streamRef.current?.getTracks().forEach(t => t.stop());
    }
  };

  const endSession = () => {
    sseRef.current?.close();
    pcRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    pcRef.current  = null;
    streamRef.current = null;
    setStatus("ended");
    setRisk(null);
    setBanners([]);
  };

  // Cleanup on unmount
  useEffect(() => () => endSession(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const scoreColor = risk ? (RISK_COLORS[risk.state] || "#22c55e") : "#22c55e";
  const scoreWidth = risk ? Math.min(risk.score, 100) : 0;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between border-b" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
        <div>
          <h1 className="text-xl font-bold">AI Proctor</h1>
          {deviceLabel && <p className="text-sm" style={{ color: "var(--muted)" }}>{deviceLabel}</p>}
        </div>
        <div className="flex items-center gap-3">
          {status === "connected" && (
            <span className="flex items-center gap-2 text-sm font-medium" style={{ color: "#22c55e" }}>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              Live
            </span>
          )}
          {status === "idle" && (
            <button
              onClick={startSession}
              className="px-5 py-2 rounded-lg text-white font-semibold transition-all hover:opacity-90"
              style={{ background: "#2563eb" }}
            >
              Start Exam
            </button>
          )}
          {status === "connecting" && (
            <button disabled className="px-5 py-2 rounded-lg text-white font-semibold opacity-60" style={{ background: "#2563eb" }}>
              Connecting…
            </button>
          )}
          {status === "connected" && (
            <button
              onClick={endSession}
              className="px-5 py-2 rounded-lg font-semibold transition-all hover:opacity-90"
              style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--foreground)" }}
            >
              End Session
            </button>
          )}
        </div>
      </header>

      <div className="flex flex-1 gap-0 overflow-hidden">

        {/* Left: video + banners + exam content */}
        <div className="flex-1 flex flex-col items-center p-6 gap-4 overflow-y-auto">

          {error && (
            <div className="w-full max-w-2xl px-4 py-3 rounded-lg text-sm border border-red-500 bg-red-950/40 text-red-300">
              {error}
            </div>
          )}

          {status === "ended" && (
            <div className="w-full max-w-2xl px-6 py-8 rounded-xl text-center" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <p className="text-2xl font-bold mb-2">Session Ended</p>
              <p style={{ color: "var(--muted)" }}>Your exam session has been closed.</p>
              <button
                onClick={() => {
                    setStatus("idle"); setPcId(null); setLabel("");
                    setTabSwitchCount(0);
                    tabSwitchCountRef.current   = 0;
                    terminationShownRef.current = false;
                  }}
                className="mt-6 px-6 py-2 rounded-lg font-medium"
                style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}
              >
                Start New Session
              </button>
            </div>
          )}

          {/* Camera feed */}
          <div className="relative w-full max-w-2xl rounded-xl overflow-hidden flex-shrink-0" style={{ background: "#000", aspectRatio: "16/9" }}>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
            />
            {status === "idle" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p style={{ color: "var(--muted)" }}>Camera will appear here</p>
              </div>
            )}
          </div>

          {/* Alert/Warning banners — stacked below video */}
          <div className="w-full max-w-2xl space-y-2">
            {banners.map(b => (
              <div
                key={b.id}
                className="px-4 py-3 rounded-lg text-sm font-medium flex items-start gap-2 animate-in fade-in slide-in-from-bottom-2 duration-300"
                style={{
                  background   : b.type === "alert" ? "rgba(239,68,68,0.15)" : "rgba(245,158,11,0.15)",
                  border       : `1px solid ${b.type === "alert" ? "#ef4444" : "#f59e0b"}`,
                  borderLeft   : `4px solid ${b.type === "alert" ? "#ef4444" : "#f59e0b"}`,
                  color        : b.type === "alert" ? "#fca5a5" : "#fcd34d",
                }}
              >
                <span className="mt-0.5 flex-shrink-0">{b.type === "alert" ? "⚠" : "ℹ"}</span>
                <span>{b.message}</span>
              </div>
            ))}
          </div>

          {/* ── Demo exam content — shown only during active session ── */}
          {status === "connected" && (
            <div className="w-full max-w-2xl space-y-4">

              {/* Question card — text is non-selectable and copy-blocked */}
              <div
                className="rounded-xl p-5 select-none"
                style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                onContextMenu={e => e.preventDefault()}
              >
                <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "var(--muted)" }}>
                  Question 1 of 3
                </p>
                <p className="text-sm leading-relaxed" style={{ color: "var(--foreground)" }}>
                  Explain the concept of <strong>Big-O notation</strong> and describe the time complexity of the following operations in a hash table: insertion, lookup, and deletion. Under what conditions does a hash table degrade to O(n) performance, and how can this be mitigated?
                </p>
                <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
                  Marks: 10 &nbsp;|&nbsp; Suggested time: 8 minutes
                </p>
              </div>

              {/* Answer textarea — paste is blocked */}
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--muted)" }}>
                  Your Answer
                </p>
                <textarea
                  className="w-full rounded-xl p-4 text-sm resize-none focus:outline-none"
                  rows={7}
                  placeholder="Type your answer here…  (copy and paste are disabled during the exam)"
                  onPaste={e => e.preventDefault()}
                  onCopy={e => e.preventDefault()}
                  onCut={e => e.preventDefault()}
                  style={{
                    background  : "var(--surface)",
                    border      : "1px solid var(--border)",
                    color       : "var(--foreground)",
                    lineHeight  : "1.6",
                  }}
                />
              </div>

            </div>
          )}
        </div>

        {/* Risk panel — right side when connected */}
        {(status === "connected" || status === "ended") && risk && (
          <div className="w-64 p-5 border-l flex flex-col gap-5 flex-shrink-0" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "var(--muted)" }}>Risk Score</p>
              <p className="text-5xl font-bold tabular-nums" style={{ color: scoreColor }}>
                {Math.round(risk.score)}
              </p>
              <p className="mt-1 text-sm font-semibold" style={{ color: scoreColor }}>
                {risk.state.replace("_", " ")}
              </p>
              <div className="mt-3 h-2 rounded-full overflow-hidden" style={{ background: "var(--surface2)" }}>
                <div
                  className="h-2 rounded-full transition-all duration-700"
                  style={{ width: `${scoreWidth}%`, background: scoreColor }}
                />
              </div>
              <div className="mt-2 flex justify-between text-xs" style={{ color: "var(--muted)" }}>
                <span>Fixed: {Math.round(risk.fixed)}</span>
                <span>Decay: {Math.round(risk.decaying)}</span>
              </div>
            </div>

            {risk.terminated && (
              <div className="px-3 py-2 rounded-lg text-sm font-bold text-center text-red-300 border border-red-800 bg-red-950/40">
                EXAM TERMINATED
              </div>
            )}

            {tabSwitchCount > 0 && (
              <div className="px-3 py-2 rounded-lg text-xs font-semibold border"
                style={{ background: "rgba(245,158,11,0.1)", borderColor: "#f59e0b", color: "#fcd34d" }}>
                Focus violations: {tabSwitchCount} / 3
                {tabSwitchCount >= 2 && <span className="block mt-0.5 font-normal">Next violation will terminate your exam.</span>}
              </div>
            )}

            <div>
              <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--muted)" }}>Rules</p>
              <ul className="text-xs space-y-1" style={{ color: "var(--muted)" }}>
                <li>• Keep your face visible at all times</li>
                <li>• Look at the screen, not away</li>
                <li>• Ensure no other person is present</li>
                <li>• No phone or notes on desk</li>
                <li>• Do not switch tabs or minimise</li>
                <li>• Do not use split screen</li>
                <li>• Copy / paste is disabled</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
