import type { SessionInfo, DetectionConfig, SessionReport, ReportMeta, RiskInfo, AlertEntry, WarningEntry, MetricsSnapshot, SystemReport, AdminSettings } from "./types";

const BASE_1 = process.env.NEXT_PUBLIC_BACKEND_URL  || "http://localhost:8000";
const BASE_2 = process.env.NEXT_PUBLIC_BACKEND_URL_2 || "";

// All backend URLs in order. Components that need multi-backend support read this.
export const BACKEND_URLS: string[] = [BASE_1, ...(BASE_2 ? [BASE_2] : [])];

export function createApi(base: string) {
  return {
    base,

    // ── Session ──────────────────────────────────────────────────────────────
    async sessions(): Promise<SessionInfo[]> {
      const r = await fetch(`${base}/sessions`);
      if (!r.ok) return [];
      return r.json();
    },

    async risk(pcId: string): Promise<RiskInfo> {
      const r = await fetch(`${base}/risk/${pcId}`);
      if (!r.ok) throw new Error("Session not found");
      return r.json();
    },

    async snapshot(pcId: string): Promise<string> {
      return `${base}/snapshot/${pcId}?t=${Date.now()}`;
    },

    snapshotUrl(pcId: string): string {
      return `${base}/snapshot/${pcId}`;
    },

    // ── WebRTC ───────────────────────────────────────────────────────────────
    async sendIceCandidate(pcId: string, candidate: RTCIceCandidateInit): Promise<void> {
      await fetch(`${base}/ice-candidate/${pcId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(candidate),
      });
    },

    async offer(sdp: string, type: string, detectionConfig?: Partial<DetectionConfig>) {
      const r = await fetch(`${base}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp, type, detection_config: detectionConfig ?? {} }),
      });
      return r.json();
    },

    // ── Exam config ───────────────────────────────────────────────────────────
    async getExamConfig(): Promise<DetectionConfig> {
      const r = await fetch(`${base}/exam/config`);
      if (!r.ok) throw new Error("Failed to get config");
      return r.json();
    },

    async setExamConfig(config: Partial<DetectionConfig>): Promise<DetectionConfig> {
      const r = await fetch(`${base}/exam/config`, {
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
      const r = await fetch(`${base}/reports`);
      if (!r.ok) return [];
      return r.json();
    },

    async listReportsMeta(): Promise<ReportMeta[]> {
      const r = await fetch(`${base}/reports/meta`);
      if (!r.ok) return [];
      return r.json();
    },

    async getReport(reportId: string): Promise<SessionReport> {
      const r = await fetch(`${base}/report/${reportId}`);
      if (!r.ok) throw new Error("Report not found");
      return r.json();
    },

    async deleteReport(reportId: string): Promise<void> {
      const r = await fetch(`${base}/report/${reportId}`, { method: "DELETE" });
      if (!r.ok) throw new Error("Delete failed");
    },

    // ── Proof ─────────────────────────────────────────────────────────────────
    proofUrl(path: string): string {
      if (path.startsWith("/proof/")) return `${base}${path}`;
      return `${base}/proof/${path}`;
    },

    // ── Session log ───────────────────────────────────────────────────────────
    async sessionLog(pcId: string): Promise<{ alert_log: AlertEntry[]; warning_log: WarningEntry[]; risk: RiskInfo }> {
      const r = await fetch(`${base}/session/${pcId}/log`);
      if (!r.ok) throw new Error("Session not found");
      return r.json();
    },

    // ── Tab switch ────────────────────────────────────────────────────────────
    async tabSwitch(pcId: string): Promise<{ ok: boolean; risk: RiskInfo }> {
      // keepalive: true — browser completes the request even when the tab is hidden.
      const r = await fetch(`${base}/tab_switch/${pcId}`, { method: "POST", keepalive: true });
      if (!r.ok) throw new Error("tab_switch failed");
      return r.json();
    },

    // ── Metrics & system report ───────────────────────────────────────────────
    async getMetrics(): Promise<MetricsSnapshot> {
      const r = await fetch(`${base}/metrics`);
      if (!r.ok) throw new Error("metrics unavailable");
      return r.json();
    },

    async getSystemReport(): Promise<SystemReport> {
      const r = await fetch(`${base}/system/report`);
      if (!r.ok) throw new Error("system report unavailable");
      return r.json();
    },

    // ── Admin settings ────────────────────────────────────────────────────────
    async getAdminSettings(): Promise<AdminSettings> {
      const r = await fetch(`${base}/admin/settings`);
      if (!r.ok) throw new Error("settings unavailable");
      return r.json();
    },

    async saveAdminSettings(patch: Partial<AdminSettings>): Promise<{ ok: boolean; changed: Partial<AdminSettings> }> {
      const r = await fetch(`${base}/admin/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!r.ok) throw new Error("settings update failed");
      return r.json();
    },

    // ── Debug overlay ─────────────────────────────────────────────────────────
    async toggleDebug(pcId: string, enabled: boolean): Promise<{ ok: boolean; debug: boolean }> {
      const r = await fetch(`${base}/debug/${pcId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!r.ok) throw new Error("debug toggle failed");
      return r.json();
    },

    // ── SSE stream ────────────────────────────────────────────────────────────
    streamUrl(pcId: string): string {
      return `${base}/stream/${pcId}`;
    },
  };
}

// Default api instance — used anywhere a single backend is sufficient.
export const api = createApi(BASE_1);

/**
 * Checks /capacity on every configured backend in parallel and returns the URL
 * with the most free session slots (least loaded).
 * Throws a user-friendly error if all backends are at capacity or unreachable.
 */
export async function pickLeastLoadedBackend(): Promise<string> {
  const results = await Promise.allSettled(
    BACKEND_URLS.map(async url => {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 3000);
      try {
        const r = await fetch(`${url}/capacity`, { signal: controller.signal });
        if (!r.ok) throw new Error("error");
        const data = await r.json() as { active: number; max: number; available: boolean };
        return { url, active: data.active, available: data.available };
      } finally {
        clearTimeout(t);
      }
    })
  );

  const available = results
    .filter((r): r is PromiseFulfilledResult<{ url: string; active: number; available: boolean }> =>
      r.status === "fulfilled" && r.value.available
    )
    .map(r => r.value)
    .sort((a, b) => a.active - b.active); // least loaded first

  if (available.length === 0) {
    throw new Error("All exam servers are currently full. Please try again in a few minutes.");
  }

  return available[0].url;
}
