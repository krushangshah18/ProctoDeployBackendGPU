export type RiskState = "NORMAL" | "WARNING" | "HIGH_RISK" | "ADMIN_REVIEW" | "TERMINATED";

export interface RiskInfo {
  score: number;
  state: RiskState;
  fixed: number;
  decaying: number;
  terminated: boolean;
}

export interface AlertEntry {
  time: string;
  elapsed_s: number;
  message: string;
  score_added?: number;
  proof?: string;
  proof_url?: string;
  proof_type?: "image" | "audio";
}

export interface WarningEntry {
  time: string;
  elapsed_s: number;
  message: string;
}

export interface SSEEvent {
  type: "connected" | "alert" | "warning" | "session_end" | "error";
  pc_id?: string;
  key?: string;
  message?: string;
  time?: string;
  elapsed_s?: number;
  score_added?: number;
  risk?: RiskInfo;
  proof_url?: string;
  proof_type?: "image" | "audio";
  terminated?: boolean;
  report_id?: string;
}

export interface SessionInfo {
  pc_id: string;
  label: string;
  connection_state: string;
  fps?: number;
  resolution?: string;
  risk_score?: number;
  risk_state?: RiskState;
  alert_count?: number;
  warning_count?: number;
  terminated?: boolean;
}

export interface DetectionConfig {
  DETECT_LOOKING_AWAY: boolean;
  DETECT_LOOKING_DOWN: boolean;
  DETECT_LOOKING_UP: boolean;
  DETECT_LOOKING_SIDE: boolean;
  DETECT_FACE_HIDDEN: boolean;
  DETECT_PARTIAL_FACE: boolean;
  DETECT_FAKE_PRESENCE: boolean;
  DETECT_SPEAKER_AUDIO: boolean;
  DETECT_PHONE: boolean;
  DETECT_BOOK: boolean;
  DETECT_HEADPHONE: boolean;
  DETECT_EARBUD: boolean;
  DETECT_MULTIPLE_PEOPLE: boolean;
}

// ── Metrics / System Report ───────────────────────────────────────────────────

export interface EndpointMetric {
  count:       number;
  errors:      number;
  lat_avg_ms:  number;
  lat_p50_ms:  number;
  lat_p95_ms:  number;
  lat_p99_ms:  number;
}

export interface MetricsSnapshot {
  generated_at : string;
  uptime_s     : number;
  uptime       : string;
  requests: {
    total         : number;
    errors        : number;
    error_rate_pct: number;
    by_endpoint   : Record<string, EndpointMetric>;
  };
  sessions: {
    total_created: number;
    active       : number;
  };
  events: {
    alerts_total  : number;
    warnings_total: number;
  };
  yolo: {
    samples      : number;
    lat_avg_ms   : number;
    lat_p95_ms   : number;
    lat_p99_ms   : number;
    lat_max_ms   : number;
  };
  coordinator: {
    tick_avg_ms    : number;
    tick_p95_ms    : number;
    tick_max_ms    : number;
    active_sessions?: number;
    max_sessions?   : number;
    tick_rate_target?: number;
    last_tick_ms?   : number;
    total_ticks?    : number;
    device?         : string;
    device_info?    : Record<string, unknown>;
  };
  system: {
    cpu_percent       : number;
    mem_rss_mb        : number;
    gpu_util_pct      : number;
    gpu_mem_used_mb   : number;
    gpu_mem_total_mb  : number;
    gpu_mem_used_pct  : number;
  };
}

export interface SystemReport extends MetricsSnapshot {
  environment: {
    python_version: string;
    platform      : string;
    hostname      : string;
    pid           : number;
    cpu_count     : number;
    cpu_phys      : number;
    ram_total_gb  : number;
    ram_avail_gb  : number;
  };
  gpu: {
    cuda_available   : boolean;
    name?            : string;
    vram_total_gb?   : number;
    vram_alloc_mb?   : number;
    vram_reserved_mb?: number;
    compute_cap?     : string;
    torch_version?   : string;
  };
  detector: {
    model_enabled    : boolean;
    device?          : string;
    half_precision?  : boolean;
    batch_supported? : boolean;
    total_batches?   : number;
    total_frames?    : number;
    last_batch_ms?   : number;
    gpu_name?        : string;
    gpu_vram_gb?     : number;
  };
  yolo_performance: MetricsSnapshot["yolo"];
  system_resources: MetricsSnapshot["system"];
  config: Record<string, unknown>;
}

export interface SessionReport {
  session_id: string;
  report_id: string;
  session_start: string;
  session_end: string;
  duration_s: number;
  total_api_alerts: number;
  total_warnings: number;
  alert_summary: Record<string, number>;
  warning_summary: Record<string, number>;
  alert_log: AlertEntry[];
  warning_log: WarningEntry[];
  risk: {
    final_score: number;
    final_state: RiskState;
    peak_score: number;
    terminated: boolean;
  };
}
