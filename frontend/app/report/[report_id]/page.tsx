"use client";

import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { SessionReport, AlertEntry, WarningEntry } from "@/lib/types";

const RISK_COLORS: Record<string, string> = {
  NORMAL      : "#22c55e",
  WARNING     : "#f59e0b",
  HIGH_RISK   : "#ef4444",
  ADMIN_REVIEW: "#dc2626",
  TERMINATED  : "#ef4444",
};

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl p-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <p className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--muted)" }}>{label}</p>
      <p className="text-3xl font-bold" style={{ color: color ?? "var(--foreground)" }}>{value}</p>
      {sub && <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>{sub}</p>}
    </div>
  );
}

function AlertLogRow({ entry, index }: { entry: AlertEntry; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-lg overflow-hidden" style={{ background: "var(--surface2)", border: "1px solid #991b1b33" }}>
      <button className="w-full px-4 py-3 flex items-start gap-4 text-left" onClick={() => setExpanded(e => !e)}>
        <span className="text-xs font-mono w-8 text-right flex-shrink-0 mt-0.5" style={{ color: "var(--muted)" }}>
          {index + 1}
        </span>
        <span className="text-xs font-mono flex-shrink-0 mt-0.5 w-14" style={{ color: "var(--muted)" }}>{entry.time}</span>
        <span className="flex-1 text-sm text-red-300 text-left">{entry.message}</span>
        {entry.score_added != null && (
          <span className="text-xs font-bold px-2 py-0.5 rounded flex-shrink-0" style={{ background: "#450a0a", color: "#f87171" }}>
            +{entry.score_added}
          </span>
        )}
        {entry.proof_url && (
          <span className="text-xs flex-shrink-0" style={{ color: "var(--muted)" }}>{expanded ? "▲" : "▼"}</span>
        )}
      </button>
      {expanded && entry.proof_url && (
        <div className="px-12 pb-4">
          {entry.proof_type === "audio" ? (
            <audio controls className="w-full">
              <source src={`${BASE}${entry.proof_url}`} type="audio/wav" />
            </audio>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={`${BASE}${entry.proof_url}`}
              alt="Proof"
              className="rounded-lg max-h-48 object-cover"
              style={{ border: "1px solid var(--border)" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

function WarnLogRow({ entry, index }: { entry: WarningEntry; index: number }) {
  return (
    <div className="px-4 py-3 rounded-lg flex items-start gap-4" style={{ background: "var(--surface2)", border: "1px solid #92400e33" }}>
      <span className="text-xs font-mono w-8 text-right flex-shrink-0" style={{ color: "var(--muted)" }}>{index + 1}</span>
      <span className="text-xs font-mono flex-shrink-0 w-14" style={{ color: "var(--muted)" }}>{entry.time}</span>
      <span className="text-sm text-amber-300">{entry.message}</span>
    </div>
  );
}

function formatDuration(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

export default function ReportPage({ params }: { params: Promise<{ report_id: string }> }) {
  const { report_id } = use(params);

  const [report, setReport]   = useState<SessionReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");
  const [tab, setTab]         = useState<"alerts" | "warnings" | "summary">("summary");

  useEffect(() => {
    api.getReport(report_id)
      .then(r => { setReport(r); setLoading(false); })
      .catch(() => { setError("Report not found"); setLoading(false); });
  }, [report_id]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--background)" }}>
        <p style={{ color: "var(--muted)" }}>Loading report…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4" style={{ background: "var(--background)" }}>
        <p className="text-xl font-bold text-red-400">{error || "Report not found"}</p>
        <Link href="/reports" style={{ color: "#2563eb" }}>← Back to Reports</Link>
      </div>
    );
  }

  const riskColor  = RISK_COLORS[report.risk?.final_state ?? "NORMAL"] ?? "#22c55e";
  const scoreWidth = Math.min(report.risk?.final_score ?? 0, 100);

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="px-6 py-4 border-b" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="max-w-5xl mx-auto flex items-center gap-4">
          <Link href="/reports" className="text-sm" style={{ color: "var(--muted)" }}>← Reports</Link>
          <div className="flex-1">
            <h1 className="text-xl font-bold">{report.session_id}</h1>
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              {report.session_start} → {report.session_end} · {formatDuration(report.duration_s)}
            </p>
          </div>
          {report.risk?.terminated && (
            <span className="text-xs font-bold px-3 py-1.5 rounded bg-red-950 text-red-400 border border-red-900">
              TERMINATED
            </span>
          )}
        </div>
      </header>

      <main className="max-w-5xl mx-auto p-6 space-y-6">

        {/* Stat cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard
            label="Final Score"
            value={Math.round(report.risk?.final_score ?? 0)}
            sub={report.risk?.final_state?.replace("_", " ")}
            color={riskColor}
          />
          <StatCard
            label="Peak Score"
            value={Math.round(report.risk?.peak_score ?? 0)}
            color={RISK_COLORS[report.risk?.final_state ?? "NORMAL"]}
          />
          <StatCard label="Alerts" value={report.total_api_alerts} color="#ef4444" />
          <StatCard label="Warnings" value={report.total_warnings} color="#f59e0b" />
        </div>

        {/* Score bar */}
        <div className="rounded-xl p-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold" style={{ color: "var(--muted)" }}>Risk Progress</span>
            <span className="text-sm font-bold" style={{ color: riskColor }}>
              {report.risk?.final_state?.replace("_", " ")}
            </span>
          </div>
          <div className="h-3 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
            <div className="h-3 rounded-full" style={{ width: `${scoreWidth}%`, background: riskColor, transition: "width 1s" }} />
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b flex" style={{ borderColor: "var(--border)" }}>
          {(["summary", "alerts", "warnings"] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-5 py-3 text-sm font-semibold capitalize border-b-2 transition-colors"
              style={{
                borderColor: tab === t ? "#2563eb" : "transparent",
                color      : tab === t ? "#2563eb" : "var(--muted)",
              }}
            >
              {t}
              {t !== "summary" && (
                <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full" style={{ background: "var(--surface2)" }}>
                  {t === "alerts" ? report.total_api_alerts : report.total_warnings}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Summary tab */}
        {tab === "summary" && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {/* Alert summary */}
            <div className="rounded-xl p-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <h3 className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: "var(--muted)" }}>Alert Summary</h3>
              {Object.keys(report.alert_summary).length === 0
                ? <p className="text-sm" style={{ color: "var(--muted)" }}>No alerts</p>
                : (
                  <div className="space-y-2">
                    {Object.entries(report.alert_summary)
                      .sort((a, b) => b[1] - a[1])
                      .map(([msg, count]) => (
                        <div key={msg} className="flex items-center justify-between gap-2">
                          <span className="text-sm text-red-300 truncate">{msg}</span>
                          <span className="text-xs font-bold px-2 py-0.5 rounded flex-shrink-0"
                            style={{ background: "#450a0a", color: "#f87171" }}>
                            ×{count}
                          </span>
                        </div>
                      ))}
                  </div>
                )}
            </div>

            {/* Warning summary */}
            <div className="rounded-xl p-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <h3 className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: "var(--muted)" }}>Warning Summary</h3>
              {Object.keys(report.warning_summary).length === 0
                ? <p className="text-sm" style={{ color: "var(--muted)" }}>No warnings</p>
                : (
                  <div className="space-y-2">
                    {Object.entries(report.warning_summary)
                      .sort((a, b) => b[1] - a[1])
                      .map(([msg, count]) => (
                        <div key={msg} className="flex items-center justify-between gap-2">
                          <span className="text-sm text-amber-300 truncate">{msg}</span>
                          <span className="text-xs font-bold px-2 py-0.5 rounded flex-shrink-0"
                            style={{ background: "#451a03", color: "#fcd34d" }}>
                            ×{count}
                          </span>
                        </div>
                      ))}
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Alert log tab */}
        {tab === "alerts" && (
          <div className="space-y-2">
            {report.alert_log.length === 0
              ? <p className="text-sm text-center py-8" style={{ color: "var(--muted)" }}>No alerts recorded</p>
              : report.alert_log.map((entry, i) => <AlertLogRow key={i} entry={entry} index={i} />)
            }
          </div>
        )}

        {/* Warning log tab */}
        {tab === "warnings" && (
          <div className="space-y-2">
            {report.warning_log.length === 0
              ? <p className="text-sm text-center py-8" style={{ color: "var(--muted)" }}>No warnings recorded</p>
              : report.warning_log.map((entry, i) => <WarnLogRow key={i} entry={entry} index={i} />)
            }
          </div>
        )}
      </main>
    </div>
  );
}
