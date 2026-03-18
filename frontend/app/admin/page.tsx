"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { SessionInfo, DetectionConfig } from "@/lib/types";

const RISK_COLORS: Record<string, { text: string; bg: string; border: string }> = {
  NORMAL      : { text: "#22c55e", bg: "#052e16", border: "#166534" },
  WARNING     : { text: "#f59e0b", bg: "#451a03", border: "#92400e" },
  HIGH_RISK   : { text: "#ef4444", bg: "#450a0a", border: "#991b1b" },
  ADMIN_REVIEW: { text: "#dc2626", bg: "#450a0a", border: "#7f1d1d" },
  TERMINATED  : { text: "#ef4444", bg: "#1c0505", border: "#7f1d1d" },
};

const DETECTION_LABELS: Record<string, string> = {
  DETECT_LOOKING_AWAY   : "Looking Away",
  DETECT_LOOKING_DOWN   : "Looking Down",
  DETECT_LOOKING_UP     : "Looking Up",
  DETECT_LOOKING_SIDE   : "Gaze Sideways",
  DETECT_FACE_HIDDEN    : "Face Hidden",
  DETECT_PARTIAL_FACE   : "Partial Face",
  DETECT_FAKE_PRESENCE  : "Fake Presence",
  DETECT_SPEAKER_AUDIO  : "Speaker Audio",
  DETECT_PHONE          : "Phone",
  DETECT_BOOK           : "Book / Notes",
  DETECT_HEADPHONE      : "Headphones",
  DETECT_EARBUD         : "Earbuds",
  DETECT_MULTIPLE_PEOPLE: "Multiple People",
};

const DETECTION_GROUPS: { label: string; keys: (keyof DetectionConfig)[] }[] = [
  { label: "Head & Gaze", keys: ["DETECT_LOOKING_AWAY", "DETECT_LOOKING_DOWN", "DETECT_LOOKING_UP", "DETECT_LOOKING_SIDE"] },
  { label: "Face",        keys: ["DETECT_FACE_HIDDEN", "DETECT_PARTIAL_FACE", "DETECT_FAKE_PRESENCE"] },
  { label: "Objects",     keys: ["DETECT_PHONE", "DETECT_BOOK", "DETECT_HEADPHONE", "DETECT_EARBUD"] },
  { label: "People & Audio", keys: ["DETECT_MULTIPLE_PEOPLE", "DETECT_SPEAKER_AUDIO"] },
];

function ScoreBar({ score, state }: { score: number; state: string }) {
  const col = RISK_COLORS[state] ?? RISK_COLORS.NORMAL;
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-bold" style={{ color: col.text }}>{Math.round(score)}</span>
        <span className="text-xs font-semibold" style={{ color: col.text }}>{state?.replace("_", " ")}</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
        <div
          className="h-1.5 rounded-full transition-all duration-700"
          style={{ width: `${Math.min(score, 100)}%`, background: col.text }}
        />
      </div>
    </div>
  );
}

function SessionCard({ s }: { s: SessionInfo }) {
  const state = s.risk_state ?? "NORMAL";
  const col   = RISK_COLORS[state] ?? RISK_COLORS.NORMAL;
  const score = s.risk_score ?? 0;

  return (
    <div
      className="rounded-xl p-5 flex flex-col gap-4 transition-all hover:scale-[1.01] hover:shadow-xl"
      style={{ background: "var(--surface)", border: `1px solid ${col.border}`, boxShadow: `0 0 0 1px ${col.border}22` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-bold text-base">{s.label}</p>
          <span className="text-xs px-2 py-0.5 rounded-full font-medium mt-1 inline-block"
            style={{ background: s.connection_state === "connected" ? "#052e16" : "var(--surface2)", color: s.connection_state === "connected" ? "#22c55e" : "var(--muted)", border: `1px solid ${s.connection_state === "connected" ? "#166534" : "var(--border)"}` }}>
            {s.connection_state}
          </span>
        </div>
        {s.terminated && (
          <span className="text-xs font-bold px-2 py-1 rounded bg-red-950 text-red-400 border border-red-900">TERMINATED</span>
        )}
      </div>

      {/* Risk bar */}
      <ScoreBar score={score} state={state} />

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          { label: "Alerts",   value: s.alert_count   ?? 0, color: "#ef4444" },
          { label: "Warnings", value: s.warning_count  ?? 0, color: "#f59e0b" },
          { label: "FPS",      value: s.fps != null ? `${s.fps}` : "—", color: "var(--muted)" },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg py-2" style={{ background: "var(--surface2)" }}>
            <p className="text-sm font-bold" style={{ color }}>{value}</p>
            <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>{label}</p>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Link
          href={`/admin/session/${s.pc_id}`}
          className="flex-1 text-center py-2 rounded-lg text-sm font-semibold transition-all hover:opacity-90"
          style={{ background: "#1d4ed8", color: "#fff" }}
        >
          View Details
        </Link>
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between gap-3 py-1.5 cursor-pointer group">
      <span className="text-sm" style={{ color: checked ? "var(--foreground)" : "var(--muted)" }}>{label}</span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="relative w-10 h-5 rounded-full transition-colors duration-200 flex-shrink-0"
        style={{ background: checked ? "#2563eb" : "var(--surface2)", border: "1px solid var(--border)" }}
      >
        <span
          className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all duration-200"
          style={{ left: checked ? "calc(100% - 18px)" : "2px" }}
        />
      </button>
    </label>
  );
}

export default function AdminPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [config, setConfig]     = useState<DetectionConfig | null>(null);
  const [saving, setSaving]     = useState(false);
  const [configOpen, setConfigOpen] = useState(false);

  const loadSessions = useCallback(async () => {
    try { setSessions(await api.sessions()); } catch { /* ignore */ }
  }, []);

  const loadConfig = useCallback(async () => {
    try { setConfig(await api.getExamConfig()); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadSessions();
    loadConfig();
    const t = setInterval(loadSessions, 3000);
    return () => clearInterval(t);
  }, [loadSessions, loadConfig]);

  const handleToggle = async (key: keyof DetectionConfig, value: boolean) => {
    if (!config) return;
    const next = { ...config, [key]: value };
    setConfig(next);
    setSaving(true);
    try {
      await api.setExamConfig({ [key]: value });
    } catch { /* revert on failure */
      setConfig(config);
    } finally {
      setSaving(false);
    }
  };

  // Count sessions that hit admin-review threshold
  const criticalCount = sessions.filter(s =>
    s.risk_state === "ADMIN_REVIEW" || s.risk_state === "TERMINATED" || (s.risk_score ?? 0) >= 100
  ).length;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="px-6 py-4 border-b flex items-center justify-between" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center gap-4">
          <Link href="/" className="text-sm" style={{ color: "var(--muted)" }}>← Home</Link>
          <h1 className="text-xl font-bold">Admin Dashboard</h1>
          {criticalCount > 0 && (
            <span className="text-xs font-bold px-2 py-1 rounded-full bg-red-900 text-red-300 animate-pulse">
              {criticalCount} need attention
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <Link href="/reports" className="text-sm font-medium px-4 py-2 rounded-lg" style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
            Reports
          </Link>
          <Link href="/admin/monitor" className="text-sm font-medium px-4 py-2 rounded-lg flex items-center gap-1.5" style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Monitor
          </Link>
          <button
            onClick={() => setConfigOpen(o => !o)}
            className="text-sm font-medium px-4 py-2 rounded-lg flex items-center gap-2"
            style={{ background: configOpen ? "#1d4ed8" : "var(--surface2)", border: "1px solid var(--border)", color: configOpen ? "#fff" : "var(--foreground)" }}
          >
            ⚙ Detection Config
            {saving && <span className="w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />}
          </button>
        </div>
      </header>

      <div className="flex flex-1">

        {/* Detection config panel */}
        {configOpen && config && (
          <aside className="w-72 border-r p-5 overflow-y-auto" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest" style={{ color: "var(--muted)" }}>Detection Settings</h2>
              <span className="text-xs" style={{ color: "var(--muted)" }}>Live — affects all sessions</span>
            </div>
            <div className="space-y-5">
              {DETECTION_GROUPS.map(group => (
                <div key={group.label}>
                  <p className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: "var(--muted)" }}>{group.label}</p>
                  <div className="space-y-0.5">
                    {group.keys.map(key => (
                      <Toggle
                        key={key}
                        label={DETECTION_LABELS[key] ?? key}
                        checked={config[key]}
                        onChange={v => handleToggle(key, v)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </aside>
        )}

        {/* Main content */}
        <main className="flex-1 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold">
              Active Candidates
              <span className="ml-2 text-sm font-normal" style={{ color: "var(--muted)" }}>
                {sessions.length} session{sessions.length !== 1 ? "s" : ""}
              </span>
            </h2>
          </div>

          {sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 gap-3">
              <div className="text-5xl">👁</div>
              <p className="text-lg font-medium" style={{ color: "var(--muted)" }}>No active sessions</p>
              <p className="text-sm" style={{ color: "var(--muted)" }}>Candidates will appear here when they join the exam</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sessions
                .sort((a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0))
                .map(s => <SessionCard key={s.pc_id} s={s} />)}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
