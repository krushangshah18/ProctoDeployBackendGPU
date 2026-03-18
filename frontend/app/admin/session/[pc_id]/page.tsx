"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { use } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { SSEEvent, RiskInfo, AlertEntry, WarningEntry } from "@/lib/types";

const RISK_COLORS: Record<string, string> = {
  NORMAL      : "#22c55e",
  WARNING     : "#f59e0b",
  HIGH_RISK   : "#ef4444",
  ADMIN_REVIEW: "#dc2626",
  TERMINATED  : "#7f1d1d",
};

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function RiskBadge({ state }: { state: string }) {
  const color = RISK_COLORS[state] ?? "#22c55e";
  return (
    <span className="text-xs font-bold px-2 py-1 rounded" style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}>
      {state?.replace("_", " ")}
    </span>
  );
}

function ProofMedia({ url, type }: { url: string; type?: string }) {
  if (type === "audio") {
    return (
      <audio controls className="w-full mt-2 h-8" style={{ filter: "invert(1) hue-rotate(180deg)" }}>
        <source src={`${BASE}${url}`} type="audio/wav" />
      </audio>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`${BASE}${url}`}
      alt="Proof"
      className="mt-2 rounded-lg w-full object-cover"
      style={{ maxHeight: 180, border: "1px solid var(--border)" }}
    />
  );
}

function AlertRow({ entry, index }: { entry: AlertEntry; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--surface2)", border: "1px solid #991b1b44" }}
    >
      <button
        className="w-full px-4 py-3 flex items-start gap-3 text-left"
        onClick={() => setExpanded(e => !e)}
      >
        <span className="text-xs font-mono mt-0.5 flex-shrink-0" style={{ color: "var(--muted)" }}>
          {String(index + 1).padStart(2, "0")} · {entry.time}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-red-300">{entry.message}</p>
          {entry.score_added != null && (
            <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>+{entry.score_added} pts</p>
          )}
        </div>
        {entry.proof_url && (
          <span className="text-xs flex-shrink-0" style={{ color: "var(--muted)" }}>
            {expanded ? "▲" : "▼"} proof
          </span>
        )}
      </button>
      {expanded && entry.proof_url && (
        <div className="px-4 pb-3">
          <ProofMedia url={entry.proof_url} type={entry.proof_type} />
        </div>
      )}
    </div>
  );
}

function WarningRow({ entry, index }: { entry: WarningEntry; index: number }) {
  return (
    <div className="px-4 py-2.5 rounded-lg flex items-start gap-3"
      style={{ background: "var(--surface2)", border: "1px solid #92400e33" }}>
      <span className="text-xs font-mono mt-0.5 flex-shrink-0" style={{ color: "var(--muted)" }}>
        {String(index + 1).padStart(2, "0")} · {entry.time}
      </span>
      <p className="text-sm text-amber-300">{entry.message}</p>
    </div>
  );
}

export default function SessionDetailPage({ params }: { params: Promise<{ pc_id: string }> }) {
  const { pc_id } = use(params);

  const sseRef         = useRef<EventSource | null>(null);
  const snapshotTimer  = useRef<ReturnType<typeof setInterval> | null>(null);

  const [risk, setRisk]             = useState<RiskInfo | null>(null);
  const [alerts, setAlerts]         = useState<AlertEntry[]>([]);
  const [warnings, setWarnings]     = useState<WarningEntry[]>([]);
  const [snapUrl, setSnapUrl]       = useState<string>("");
  const [sessionEnded, setEnded]    = useState(false);
  const [reportId, setReportId]     = useState<string | null>(null);
  const [activeTab, setActiveTab]   = useState<"alerts" | "warnings">("alerts");
  const [candidateLabel, setLabel]  = useState(pc_id);
  const [debugMode, setDebugMode]   = useState(false);

  const refreshSnapshot = useCallback(() => {
    setSnapUrl(`${BASE}/snapshot/${pc_id}?t=${Date.now()}`);
  }, [pc_id]);

  const handleDebugToggle = useCallback(async () => {
    const next = !debugMode;
    setDebugMode(next);
    try {
      await api.toggleDebug(pc_id, next);
    } catch {
      setDebugMode(d => !d); // revert on error
    }
  }, [debugMode, pc_id]);

  useEffect(() => {
    // Start snapshot polling at ~7 fps
    refreshSnapshot();
    snapshotTimer.current = setInterval(refreshSnapshot, 150);

    // Pre-populate alerts/warnings from history before SSE connects
    api.sessionLog(pc_id).then(log => {
      setRisk(log.risk);
      // alert_log is oldest-first; reverse so newest is at top
      setAlerts([...log.alert_log].reverse());
      setWarnings([...log.warning_log].reverse());
    }).catch(() => {});

    // Connect SSE
    const es = new EventSource(api.streamUrl(pc_id));
    sseRef.current = es;

    es.onmessage = (e) => {
      try {
        const event: SSEEvent = JSON.parse(e.data);

        if (event.risk) setRisk(event.risk);

        if (event.type === "alert" && event.message) {
          const newEntry: AlertEntry = {
            time      : event.time ?? "",
            elapsed_s : event.elapsed_s ?? 0,
            message   : event.message!,
            score_added: event.score_added,
            proof_url  : event.proof_url,
            proof_type : event.proof_type,
          };
          setAlerts(prev => {
            // Deduplicate: skip if same elapsed_s + message already present
            const dup = prev.some(e => e.elapsed_s === newEntry.elapsed_s && e.message === newEntry.message);
            return dup ? prev : [newEntry, ...prev];
          });
        }

        if (event.type === "warning" && event.message) {
          const newWarn: WarningEntry = {
            time     : event.time ?? "",
            elapsed_s: event.elapsed_s ?? 0,
            message  : event.message!,
          };
          setWarnings(prev => {
            const dup = prev.some(e => e.elapsed_s === newWarn.elapsed_s && e.message === newWarn.message);
            return dup ? prev : [newWarn, ...prev];
          });
        }

        if (event.type === "session_end") {
          setEnded(true);
          setReportId(event.report_id ?? null);
          es.close();
          if (snapshotTimer.current) clearInterval(snapshotTimer.current);
        }
      } catch { /* ignore */ }
    };

    // Fetch initial session info for label
    api.sessions().then(list => {
      const s = list.find(s => s.pc_id === pc_id);
      if (s?.label) setLabel(s.label);
    }).catch(() => {});

    return () => {
      es.close();
      if (snapshotTimer.current) clearInterval(snapshotTimer.current);
    };
  }, [pc_id, refreshSnapshot]);

  const scoreColor = risk ? (RISK_COLORS[risk.state] ?? "#22c55e") : "#22c55e";

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="px-6 py-4 border-b flex items-center gap-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <Link href="/admin" className="text-sm" style={{ color: "var(--muted)" }}>← Dashboard</Link>
        <h1 className="text-lg font-bold">{candidateLabel}</h1>
        {risk && <RiskBadge state={risk.state} />}
        {sessionEnded && (
          <span className="text-xs px-2 py-1 rounded font-bold bg-zinc-800 text-zinc-400">Session Ended</span>
        )}
        {reportId && (
          <Link
            href={`/report/${reportId}`}
            className="ml-auto text-sm font-semibold px-4 py-2 rounded-lg"
            style={{ background: "#1d4ed8", color: "#fff" }}
          >
            View Report →
          </Link>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Left: Live snapshot + risk score */}
        <div className="w-[420px] flex-shrink-0 p-5 flex flex-col gap-4 border-r" style={{ borderColor: "var(--border)" }}>
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--muted)" }}>
                Live View {sessionEnded ? "(ended)" : ""}
              </p>
              {!sessionEnded && (
                <button
                  onClick={handleDebugToggle}
                  className="text-xs font-semibold px-3 py-1 rounded-lg transition-all"
                  style={{
                    background  : debugMode ? "rgba(99,102,241,0.2)" : "var(--surface2)",
                    border      : `1px solid ${debugMode ? "#6366f1" : "var(--border)"}`,
                    color       : debugMode ? "#a5b4fc" : "var(--muted)",
                  }}
                >
                  {debugMode ? "Debug ON" : "Debug OFF"}
                </button>
              )}
            </div>
            <div className="rounded-xl overflow-hidden" style={{ aspectRatio: "4/3", background: "#000" }}>
              {snapUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={snapUrl} alt="Live view" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-sm" style={{ color: "var(--muted)" }}>Waiting for stream…</p>
                </div>
              )}
            </div>
          </div>

          {/* Risk score card */}
          {risk && (
            <div className="rounded-xl p-4 flex flex-col gap-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--muted)" }}>Risk Score</span>
                <RiskBadge state={risk.state} />
              </div>
              <p className="text-5xl font-bold tabular-nums" style={{ color: scoreColor }}>{Math.round(risk.score)}</p>
              <div className="h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                <div className="h-2 rounded-full transition-all duration-700" style={{ width: `${Math.min(risk.score, 100)}%`, background: scoreColor }} />
              </div>
              <div className="flex justify-between text-xs" style={{ color: "var(--muted)" }}>
                <span>Fixed: {Math.round(risk.fixed)}</span>
                <span>Decaying: {Math.round(risk.decaying)}</span>
              </div>
              {risk.terminated && (
                <div className="text-sm font-bold text-center py-1.5 rounded bg-red-950 text-red-400 border border-red-900">
                  EXAM TERMINATED
                </div>
              )}
              {risk.score >= 100 && !risk.terminated && (
                <div className="text-sm font-bold text-center py-1.5 rounded bg-red-950 text-red-300 border border-red-900 animate-pulse">
                  ⚠ Score ≥ 100 — Review Required
                </div>
              )}
            </div>
          )}

          {/* Counts */}
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg py-3 text-center" style={{ background: "var(--surface)" }}>
              <p className="text-2xl font-bold text-red-400">{alerts.length}</p>
              <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>Alerts</p>
            </div>
            <div className="rounded-lg py-3 text-center" style={{ background: "var(--surface)" }}>
              <p className="text-2xl font-bold text-amber-400">{warnings.length}</p>
              <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>Warnings</p>
            </div>
          </div>
        </div>

        {/* Right: Alert / Warning log */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* Tabs */}
          <div className="flex border-b px-5" style={{ borderColor: "var(--border)" }}>
            {(["alerts", "warnings"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="px-4 py-3 text-sm font-semibold capitalize border-b-2 transition-colors"
                style={{
                  borderColor: activeTab === tab ? "#2563eb" : "transparent",
                  color      : activeTab === tab ? "#2563eb" : "var(--muted)",
                }}
              >
                {tab}
                <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full" style={{ background: "var(--surface2)" }}>
                  {tab === "alerts" ? alerts.length : warnings.length}
                </span>
              </button>
            ))}
          </div>

          {/* Log entries */}
          <div className="flex-1 overflow-y-auto p-5 space-y-2">
            {activeTab === "alerts" && (
              alerts.length === 0
                ? <p className="text-sm text-center py-12" style={{ color: "var(--muted)" }}>No alerts yet</p>
                : alerts.map((entry, i) => <AlertRow key={i} entry={entry} index={alerts.length - 1 - i} />)
            )}
            {activeTab === "warnings" && (
              warnings.length === 0
                ? <p className="text-sm text-center py-12" style={{ color: "var(--muted)" }}>No warnings yet</p>
                : warnings.map((entry, i) => <WarningRow key={i} entry={entry} index={warnings.length - 1 - i} />)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
