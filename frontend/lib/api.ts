import type { SessionInfo, DetectionConfig, SessionReport, RiskInfo, AlertEntry, WarningEntry, MetricsSnapshot, SystemReport } from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const api = {
  // ── Session ──────────────────────────────────────────────────────────────
  async sessions(): Promise<SessionInfo[]> {
    const r = await fetch(`${BASE}/sessions`);
    if (!r.ok) return [];
    return r.json();
  },

  async risk(pcId: string): Promise<RiskInfo> {
    const r = await fetch(`${BASE}/risk/${pcId}`);
    if (!r.ok) throw new Error("Session not found");
    return r.json();
  },

  async snapshot(pcId: string): Promise<string> {
    return `${BASE}/snapshot/${pcId}?t=${Date.now()}`;
  },

  snapshotUrl(pcId: string): string {
    return `${BASE}/snapshot/${pcId}`;
  },

  // ── WebRTC ───────────────────────────────────────────────────────────────
  async offer(sdp: string, type: string, detectionConfig?: Partial<DetectionConfig>) {
    const r = await fetch(`${BASE}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp, type, detection_config: detectionConfig ?? {} }),
    });
    return r.json();
  },

  // ── Exam config ───────────────────────────────────────────────────────────
  async getExamConfig(): Promise<DetectionConfig> {
    const r = await fetch(`${BASE}/exam/config`);
    if (!r.ok) throw new Error("Failed to get config");
    return r.json();
  },

  async setExamConfig(config: Partial<DetectionConfig>): Promise<DetectionConfig> {
    const r = await fetch(`${BASE}/exam/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!r.ok) throw new Error("Failed to update config");
    const { updated } = await r.json();
    return updated;
  },

  // ── Reports ───────────────────────────────────────────────────────────────
  async listReports(): Promise<string[]> {
    const r = await fetch(`${BASE}/reports`);
    if (!r.ok) return [];
    return r.json();
  },

  async getReport(reportId: string): Promise<SessionReport> {
    const r = await fetch(`${BASE}/report/${reportId}`);
    if (!r.ok) throw new Error("Report not found");
    return r.json();
  },

  // ── Proof ─────────────────────────────────────────────────────────────────
  proofUrl(path: string): string {
    if (path.startsWith("/proof/")) return `${BASE}${path}`;
    return `${BASE}/proof/${path}`;
  },

  // ── Session log (historical alerts/warnings) ──────────────────────────────
  async sessionLog(pcId: string): Promise<{ alert_log: AlertEntry[]; warning_log: WarningEntry[]; risk: RiskInfo }> {
    const r = await fetch(`${BASE}/session/${pcId}/log`);
    if (!r.ok) throw new Error("Session not found");
    return r.json();
  },

  // ── Tab switch reporting ───────────────────────────────────────────────────
  async tabSwitch(pcId: string): Promise<{ ok: boolean; risk: RiskInfo }> {
    // keepalive: true — browser completes the request even when the tab is
    // hidden/backgrounded (default fetch is suspended when tab goes hidden,
    // which is exactly when this is called).
    const r = await fetch(`${BASE}/tab_switch/${pcId}`, { method: "POST", keepalive: true });
    if (!r.ok) throw new Error("tab_switch failed");
    return r.json();
  },

  // ── Metrics & system report ───────────────────────────────────────────────
  async getMetrics(): Promise<MetricsSnapshot> {
    const r = await fetch(`${BASE}/metrics`);
    if (!r.ok) throw new Error("metrics unavailable");
    return r.json();
  },

  async getSystemReport(): Promise<SystemReport> {
    const r = await fetch(`${BASE}/system/report`);
    if (!r.ok) throw new Error("system report unavailable");
    return r.json();
  },

  // ── Debug overlay ─────────────────────────────────────────────────────────
  async toggleDebug(pcId: string, enabled: boolean): Promise<{ ok: boolean; debug: boolean }> {
    const r = await fetch(`${BASE}/debug/${pcId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) throw new Error("debug toggle failed");
    return r.json();
  },

  // ── SSE stream ────────────────────────────────────────────────────────────
  streamUrl(pcId: string): string {
    return `${BASE}/stream/${pcId}`;
  },
};
