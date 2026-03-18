"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { MetricsSnapshot, SystemReport, EndpointMetric } from "@/lib/types";

// ── Tiny helpers ──────────────────────────────────────────────────────────────

function fmt(n: number | undefined, decimals = 1) {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

function pct(used: number, total: number) {
  if (!total) return 0;
  return Math.min(100, (used / total) * 100);
}

// ── Reusable primitives ───────────────────────────────────────────────────────

function StatCard({
  label, value, sub, color = "var(--foreground)", accent,
}: {
  label: string; value: React.ReactNode; sub?: string;
  color?: string; accent?: string;
}) {
  return (
    <div className="rounded-xl p-4 flex flex-col gap-1"
      style={{ background: "var(--surface)", border: `1px solid ${accent ?? "var(--border)"}` }}>
      <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--muted)" }}>{label}</p>
      <p className="text-2xl font-bold tabular-nums" style={{ color }}>{value}</p>
      {sub && <p className="text-xs" style={{ color: "var(--muted)" }}>{sub}</p>}
    </div>
  );
}

function GaugeBar({
  label, value, max, unit = "%", color,
}: {
  label: string; value: number; max: number; unit?: string; color: string;
}) {
  const pctFill = Math.min(100, max ? (value / max) * 100 : 0);
  return (
    <div className="rounded-xl p-4 flex flex-col gap-2"
      style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="flex justify-between items-center">
        <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--muted)" }}>{label}</p>
        <p className="text-sm font-bold tabular-nums" style={{ color }}>
          {fmt(value, unit === "%" ? 1 : 0)}{unit}
        </p>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{ width: `${pctFill}%`, background: color }}
        />
      </div>
      {max !== 100 && (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          {fmt(value, 0)} / {fmt(max, 0)} {unit}
        </p>
      )}
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-bold uppercase tracking-widest pt-2" style={{ color: "var(--muted)" }}>
      {children}
    </h2>
  );
}

// ── Request table ──────────────────────────────────────────────────────────────

function RequestTable({ endpoints }: { endpoints: Record<string, EndpointMetric> }) {
  const rows = Object.entries(endpoints).sort((a, b) => b[1].count - a[1].count);
  if (!rows.length) {
    return <p className="text-sm py-6 text-center" style={{ color: "var(--muted)" }}>No requests recorded yet</p>;
  }
  return (
    <div className="overflow-x-auto rounded-xl" style={{ border: "1px solid var(--border)" }}>
      <table className="w-full text-sm">
        <thead>
          <tr style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
            {["Endpoint", "Count", "Errors", "Avg ms", "p50 ms", "p95 ms", "p99 ms"].map(h => (
              <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider"
                style={{ color: "var(--muted)" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([ep, m]) => (
            <tr key={ep} className="border-t" style={{ borderColor: "var(--border)" }}>
              <td className="px-4 py-2.5 font-mono text-xs" style={{ color: "var(--foreground)" }}>{ep}</td>
              <td className="px-4 py-2.5 tabular-nums">{m.count}</td>
              <td className="px-4 py-2.5 tabular-nums" style={{ color: m.errors > 0 ? "#ef4444" : "var(--muted)" }}>
                {m.errors > 0 ? m.errors : "—"}
              </td>
              <td className="px-4 py-2.5 tabular-nums">{fmt(m.lat_avg_ms)}</td>
              <td className="px-4 py-2.5 tabular-nums">{fmt(m.lat_p50_ms)}</td>
              <td className="px-4 py-2.5 tabular-nums"
                style={{ color: m.lat_p95_ms > 200 ? "#f59e0b" : "inherit" }}>{fmt(m.lat_p95_ms)}</td>
              <td className="px-4 py-2.5 tabular-nums"
                style={{ color: m.lat_p99_ms > 500 ? "#ef4444" : "inherit" }}>{fmt(m.lat_p99_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── System report panel ────────────────────────────────────────────────────────

function ReportPanel({ report }: { report: SystemReport }) {
  const rows: [string, string][] = [
    ["Hostname",        report.environment?.hostname ?? "—"],
    ["Platform",        report.environment?.platform ?? "—"],
    ["Python",          report.environment?.python_version ?? "—"],
    ["PID",             String(report.environment?.pid ?? "—")],
    ["CPU cores",       `${report.environment?.cpu_phys ?? "?"} physical / ${report.environment?.cpu_count ?? "?"} logical`],
    ["RAM total",       `${fmt(report.environment?.ram_total_gb, 1)} GB`],
    ["RAM available",   `${fmt(report.environment?.ram_avail_gb, 1)} GB`],
    ["CUDA available",  report.gpu?.cuda_available ? "Yes" : "No"],
    ...(report.gpu?.cuda_available ? [
      ["GPU",             report.gpu?.name ?? "—"] as [string, string],
      ["GPU VRAM",        `${fmt(report.gpu?.vram_total_gb, 1)} GB`] as [string, string],
      ["Compute cap.",    report.gpu?.compute_cap ?? "—"] as [string, string],
      ["PyTorch",         report.gpu?.torch_version ?? "—"] as [string, string],
    ] : []),
    ["Model device",    report.detector?.device ?? "—"],
    ["FP16 (half)",     report.detector?.half_precision ? "Enabled" : "Disabled"],
    ["Batch inference", report.detector?.batch_supported ? "Yes" : "No"],
    ["Total batches",   String(report.detector?.total_batches ?? "—")],
    ["Total frames",    String(report.detector?.total_frames ?? "—")],
  ];

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)" }}>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([k, v], i) => (
            <tr key={k} className={i > 0 ? "border-t" : ""} style={{ borderColor: "var(--border)" }}>
              <td className="px-4 py-2.5 text-xs font-medium w-40" style={{ color: "var(--muted)", background: "var(--surface)" }}>{k}</td>
              <td className="px-4 py-2.5 text-xs font-mono" style={{ color: "var(--foreground)" }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function MonitorPage() {
  const [metrics, setMetrics]       = useState<MetricsSnapshot | null>(null);
  const [report,  setReport]        = useState<SystemReport | null>(null);
  const [tab,     setTab]           = useState<"live" | "report">("live");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [error, setError]           = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const m = await api.getMetrics();
      setMetrics(m);
      setLastUpdated(new Date());
      setError("");
    } catch {
      setError("Backend unreachable — retrying…");
    }
  }, []);

  const fetchReport = useCallback(async () => {
    setLoadingReport(true);
    try {
      const r = await api.getSystemReport();
      setReport(r);
      setError("");
    } catch {
      setError("Could not load system report");
    } finally {
      setLoadingReport(false);
    }
  }, []);

  // Auto-refresh metrics every 3 s
  useEffect(() => {
    fetchMetrics();
    if (autoRefresh) {
      timerRef.current = setInterval(fetchMetrics, 3000);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [autoRefresh, fetchMetrics]);

  // Fetch system report when tab switches to "report"
  useEffect(() => {
    if (tab === "report" && !report) fetchReport();
  }, [tab, report, fetchReport]);

  const m = metrics;
  const sys = m?.system;
  const gpuAvail = sys && sys.gpu_mem_total_mb > 0;

  const downloadReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `proctor-system-report-${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="px-6 py-4 border-b flex items-center justify-between flex-shrink-0"
        style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center gap-4">
          <Link href="/admin" className="text-sm" style={{ color: "var(--muted)" }}>← Dashboard</Link>
          <h1 className="text-xl font-bold">System Monitor</h1>
          {lastUpdated && (
            <span className="text-xs" style={{ color: "var(--muted)" }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Auto-refresh toggle */}
          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: "var(--muted)" }}>
            <span>Auto-refresh</span>
            <button
              onClick={() => setAutoRefresh(v => !v)}
              className="relative w-9 h-5 rounded-full transition-colors"
              style={{ background: autoRefresh ? "#2563eb" : "var(--surface2)", border: "1px solid var(--border)" }}
            >
              <span className="absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white transition-all"
                style={{ left: autoRefresh ? "calc(100% - 16px)" : "2px" }} />
            </button>
          </label>
          <button onClick={fetchMetrics}
            className="text-sm px-3 py-1.5 rounded-lg font-medium"
            style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
            Refresh
          </button>
          {tab === "report" && (
            <button onClick={report ? downloadReport : fetchReport}
              disabled={loadingReport}
              className="text-sm px-3 py-1.5 rounded-lg font-medium"
              style={{ background: "#1d4ed8", color: "#fff", opacity: loadingReport ? 0.6 : 1 }}>
              {loadingReport ? "Loading…" : report ? "Download JSON" : "Load Report"}
            </button>
          )}
        </div>
      </header>

      {/* Tabs */}
      <div className="flex border-b px-6" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
        {(["live", "report"] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className="px-4 py-3 text-sm font-semibold capitalize border-b-2 transition-colors"
            style={{
              borderColor: tab === t ? "#2563eb" : "transparent",
              color      : tab === t ? "#2563eb" : "var(--muted)",
            }}>
            {t === "live" ? "Live Metrics" : "System Report"}
          </button>
        ))}
      </div>

      {error && (
        <div className="mx-6 mt-4 px-4 py-2.5 rounded-lg text-sm border border-amber-700 bg-amber-950/30 text-amber-300">
          {error}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* ── LIVE METRICS TAB ─────────────────────────────────────────────── */}
        {tab === "live" && (
          <>
            {!m ? (
              <div className="flex items-center justify-center py-24">
                <p className="text-sm" style={{ color: "var(--muted)" }}>Connecting to backend…</p>
              </div>
            ) : (
              <>
                {/* Overview stat cards */}
                <SectionTitle>Overview</SectionTitle>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  <StatCard label="Uptime"          value={m.uptime} />
                  <StatCard label="Total Requests"  value={m.requests.total.toLocaleString()} />
                  <StatCard label="Error Rate"
                    value={`${fmt(m.requests.error_rate_pct)}%`}
                    color={m.requests.error_rate_pct > 1 ? "#ef4444" : "#22c55e"}
                    accent={m.requests.error_rate_pct > 1 ? "#991b1b44" : undefined}
                  />
                  <StatCard label="Active Sessions" value={m.sessions.active}
                    color={m.sessions.active > 0 ? "#22c55e" : "var(--foreground)"} />
                  <StatCard label="Total Alerts"    value={m.events.alerts_total}
                    color={m.events.alerts_total > 0 ? "#ef4444" : "var(--muted)"} />
                  <StatCard label="Total Warnings"  value={m.events.warnings_total}
                    color={m.events.warnings_total > 0 ? "#f59e0b" : "var(--muted)"} />
                </div>

                {/* System resources */}
                <SectionTitle>System Resources</SectionTitle>
                <div className={`grid gap-3 ${gpuAvail ? "grid-cols-2 lg:grid-cols-4" : "grid-cols-2"}`}>
                  <GaugeBar label="CPU Usage"   value={sys!.cpu_percent}     max={100}  unit="%" color="#6366f1" />
                  <GaugeBar label="Memory (RSS)" value={sys!.mem_rss_mb}     max={Math.max(sys!.mem_rss_mb * 1.5, 512)} unit=" MB" color="#06b6d4" />
                  {gpuAvail && <>
                    <GaugeBar label="GPU Utilisation" value={sys!.gpu_util_pct}   max={100} unit="%" color="#a855f7" />
                    <GaugeBar label="GPU VRAM"        value={sys!.gpu_mem_used_mb} max={sys!.gpu_mem_total_mb} unit=" MB" color="#ec4899" />
                  </>}
                  {!gpuAvail && (
                    <div className="rounded-xl p-4 flex items-center justify-center col-span-0"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <p className="text-xs" style={{ color: "var(--muted)" }}>GPU: not available (CPU mode)</p>
                    </div>
                  )}
                </div>

                {/* Inference & tick performance */}
                <SectionTitle>Inference Performance</SectionTitle>
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                  <StatCard label="YOLO avg"   value={`${fmt(m.yolo.lat_avg_ms)} ms`}  sub={`${m.yolo.samples} samples`} />
                  <StatCard label="YOLO p95"   value={`${fmt(m.yolo.lat_p95_ms)} ms`}
                    color={m.yolo.lat_p95_ms > 200 ? "#f59e0b" : "var(--foreground)"} />
                  <StatCard label="YOLO p99"   value={`${fmt(m.yolo.lat_p99_ms)} ms`}
                    color={m.yolo.lat_p99_ms > 500 ? "#ef4444" : "var(--foreground)"} />
                  <StatCard label="YOLO max"   value={`${fmt(m.yolo.lat_max_ms)} ms`} />
                  <StatCard label="Tick avg"   value={`${fmt(m.coordinator.tick_avg_ms)} ms`}
                    sub={`target ${m.coordinator.tick_rate_target ?? 10} Hz`} />
                  <StatCard label="Tick p95"   value={`${fmt(m.coordinator.tick_p95_ms)} ms`}
                    color={m.coordinator.tick_p95_ms > 150 ? "#f59e0b" : "var(--foreground)"} />
                </div>

                {/* Coordinator detail */}
                {m.coordinator && (
                  <>
                    <SectionTitle>Coordinator</SectionTitle>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <StatCard label="Active Sessions"  value={`${m.coordinator.active_sessions ?? m.sessions.active} / ${m.coordinator.max_sessions ?? "—"}`} />
                      <StatCard label="Total Ticks"      value={(m.coordinator.total_ticks ?? 0).toLocaleString()} />
                      <StatCard label="Last Tick"        value={`${fmt(m.coordinator.last_tick_ms)} ms`} />
                      <StatCard label="Device"           value={m.coordinator.device ?? "—"} />
                    </div>
                  </>
                )}

                {/* Request breakdown table */}
                <SectionTitle>Request Breakdown</SectionTitle>
                <RequestTable endpoints={m.requests.by_endpoint} />
              </>
            )}
          </>
        )}

        {/* ── SYSTEM REPORT TAB ────────────────────────────────────────────── */}
        {tab === "report" && (
          <>
            {loadingReport && (
              <div className="flex items-center justify-center py-24">
                <p className="text-sm" style={{ color: "var(--muted)" }}>Loading system report…</p>
              </div>
            )}

            {!loadingReport && !report && (
              <div className="flex flex-col items-center justify-center py-24 gap-4">
                <p className="text-sm" style={{ color: "var(--muted)" }}>System report not loaded yet</p>
                <button onClick={fetchReport}
                  className="px-5 py-2 rounded-lg text-sm font-semibold text-white"
                  style={{ background: "#1d4ed8" }}>
                  Load System Report
                </button>
              </div>
            )}

            {!loadingReport && report && (
              <>
                {/* Header info */}
                <div className="flex items-center justify-between">
                  <SectionTitle>Environment & Hardware</SectionTitle>
                  <p className="text-xs" style={{ color: "var(--muted)" }}>
                    Generated: {new Date(report.generated_at).toLocaleString()}
                  </p>
                </div>
                <ReportPanel report={report} />

                {/* Perf snapshot from system report */}
                <SectionTitle>Performance Snapshot</SectionTitle>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <StatCard label="Uptime"        value={report.uptime} />
                  <StatCard label="Total Sessions" value={report.sessions?.total_created ?? "—"} />
                  <StatCard label="Total Alerts"   value={report.events?.alerts_total ?? "—"}
                    color={(report.events?.alerts_total ?? 0) > 0 ? "#ef4444" : "var(--muted)"} />
                  <StatCard label="Total Warnings" value={report.events?.warnings_total ?? "—"}
                    color={(report.events?.warnings_total ?? 0) > 0 ? "#f59e0b" : "var(--muted)"} />
                  <StatCard label="YOLO avg"       value={`${fmt(report.yolo_performance?.lat_avg_ms)} ms`} />
                  <StatCard label="YOLO p95"       value={`${fmt(report.yolo_performance?.lat_p95_ms)} ms`} />
                  <StatCard label="Total Requests" value={report.requests?.total.toLocaleString() ?? "—"} />
                  <StatCard label="Error Rate"     value={`${fmt(report.requests?.error_rate_pct)}%`}
                    color={(report.requests?.error_rate_pct ?? 0) > 1 ? "#ef4444" : "#22c55e"} />
                </div>

                {/* Request table */}
                <SectionTitle>Request Breakdown</SectionTitle>
                <RequestTable endpoints={report.requests?.by_endpoint ?? {}} />

                {/* Active sessions */}
                {Array.isArray((report as any).sessions?.active_details) && (report as any).sessions.active_details.length > 0 && (
                  <>
                    <SectionTitle>Active Sessions at Report Time</SectionTitle>
                    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                      <table className="w-full text-sm">
                        <thead>
                          <tr style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
                            {["Label", "State", "FPS", "Risk Score", "Risk State", "Alerts", "Warnings"].map(h => (
                              <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider"
                                style={{ color: "var(--muted)" }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(report as any).sessions.active_details.map((s: any) => (
                            <tr key={s.pc_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                              <td className="px-4 py-2.5 font-medium">{s.label}</td>
                              <td className="px-4 py-2.5 text-xs">{s.state}</td>
                              <td className="px-4 py-2.5 tabular-nums">{s.fps ?? "—"}</td>
                              <td className="px-4 py-2.5 tabular-nums font-bold"
                                style={{ color: s.risk_score > 60 ? "#ef4444" : s.risk_score > 30 ? "#f59e0b" : "#22c55e" }}>
                                {s.risk_score}
                              </td>
                              <td className="px-4 py-2.5 text-xs">{s.risk_state}</td>
                              <td className="px-4 py-2.5 tabular-nums" style={{ color: s.alerts > 0 ? "#ef4444" : "var(--muted)" }}>{s.alerts}</td>
                              <td className="px-4 py-2.5 tabular-nums" style={{ color: s.warnings > 0 ? "#f59e0b" : "var(--muted)" }}>{s.warnings}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {/* Config */}
                {report.config && Object.keys(report.config).length > 0 && (
                  <>
                    <SectionTitle>Active Configuration</SectionTitle>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                      {Object.entries(report.config)
                        .filter(([, v]) => typeof v !== "object")
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(([k, v]) => (
                          <div key={k} className="rounded-lg px-3 py-2 flex justify-between items-center gap-2"
                            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                            <span className="text-xs font-mono truncate" style={{ color: "var(--muted)" }}>{k}</span>
                            <span className="text-xs font-bold flex-shrink-0"
                              style={{ color: v === true ? "#22c55e" : v === false ? "#6b7280" : "var(--foreground)" }}>
                              {String(v)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
