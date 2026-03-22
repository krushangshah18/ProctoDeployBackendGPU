"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { createApi, BACKEND_URLS } from "@/lib/api";
import type { AdminSettings } from "@/lib/types";

// ── Helpers ───────────────────────────────────────────────────────────────────

function portLabel(url: string) {
  try { return `:${new URL(url).port || "80"}`; } catch { return url; }
}

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
  DETECT_BOOK           : "Book",
  DETECT_HEADPHONE      : "Headphones",
  DETECT_EARBUD         : "Earbuds",
  DETECT_MULTIPLE_PEOPLE: "Multiple People",
};

const THRESHOLD_META: Record<string, { label: string; min: number; max: number; step: number; unit?: string }> = {
  LOOKING_AWAY_THRESHOLD: { label: "Looking Away Duration",    min: 0.5,  max: 8,    step: 0.5,  unit: "s" },
  GAZE_THRESHOLD        : { label: "Gaze Duration",            min: 0.5,  max: 8,    step: 0.5,  unit: "s" },
  LOOK_AWAY_YAW         : { label: "Yaw threshold (away)",     min: 0.05, max: 0.50, step: 0.01 },
  LOOK_DOWN_PITCH       : { label: "Pitch threshold (down)",   min: 0.05, max: 0.40, step: 0.01 },
  LOOK_UP_PITCH         : { label: "Pitch threshold (up)",     min: -0.40,max: -0.02, step: 0.01 },
  GAZE_LEFT             : { label: "Gaze threshold (left)",    min: -0.40,max: -0.02, step: 0.01 },
  GAZE_RIGHT            : { label: "Gaze threshold (right)",   min: 0.02, max: 0.40, step: 0.01 },
  EAR_THRESHOLD         : { label: "Blink EAR threshold",      min: 0.10, max: 0.40, step: 0.01 },
  BLINK_MIN_DURATION_S  : { label: "Blink min duration",       min: 0.02, max: 0.30, step: 0.01, unit: "s" },
  BLINK_MAX_DURATION_S  : { label: "Blink max duration",       min: 0.15, max: 1.00, step: 0.05, unit: "s" },
  MIN_FACE_WIDTH        : { label: "Min face width",           min: 40,   max: 200,  step: 5,    unit: "px" },
  MIN_FACE_HEIGHT       : { label: "Min face height",          min: 50,   max: 250,  step: 5,    unit: "px" },
};

const OBJECT_META: Record<string, { label: string; min: number; max: number; step: number; unit?: string }> = {
  OBJECT_WINDOW      : { label: "Vote window (frames)",       min: 5,    max: 30,   step: 1 },
  PHONE_MIN_VOTES    : { label: "Phone min votes",            min: 2,    max: 20,   step: 1 },
  BOOK_MIN_VOTES     : { label: "Book min votes",             min: 2,    max: 20,   step: 1 },
  HEADPHONE_MIN_VOTES: { label: "Headphone min votes",        min: 2,    max: 20,   step: 1 },
  EARBUD_MIN_VOTES   : { label: "Earbud min votes",           min: 2,    max: 20,   step: 1 },
  OBJECT_MIN_VOTES   : { label: "Default min votes",          min: 2,    max: 15,   step: 1 },
  YOLO_DEFAULT_CONF  : { label: "YOLO default confidence",    min: 0.10, max: 0.90, step: 0.05 },
  YOLO_PHONE_CONF    : { label: "YOLO phone confidence",      min: 0.10, max: 0.90, step: 0.05 },
  YOLO_BOOK_CONF     : { label: "YOLO book confidence",       min: 0.10, max: 0.90, step: 0.05 },
  YOLO_AUDIO_CONF    : { label: "YOLO audio confidence",      min: 0.10, max: 0.90, step: 0.05 },
};

const SCORING_META: Record<string, { label: string; min: number; max: number; step: number; unit?: string }> = {
  STATE_WARNING             : { label: "Warning threshold",          min: 10,  max: 80,   step: 5,   unit: "pts" },
  STATE_HIGH_RISK           : { label: "High risk threshold",        min: 30,  max: 120,  step: 5,   unit: "pts" },
  STATE_ADMIN               : { label: "Admin review threshold",     min: 60,  max: 200,  step: 5,   unit: "pts" },
  DECAY_AMOUNT              : { label: "Score decay amount",         min: 1,   max: 30,   step: 1,   unit: "pts" },
  TAB_SWITCH_SCORE          : { label: "Tab switch score",           min: 5,   max: 50,   step: 5,   unit: "pts" },
  TAB_SWITCH_TERMINATE_COUNT: { label: "Tab switches → terminate",   min: 2,   max: 10,   step: 1 },
  GAZE_SCORE                : { label: "Gaze event score",           min: 1,   max: 30,   step: 1,   unit: "pts" },
  PHONE_SCORE_2ND           : { label: "Phone score (2nd+)",         min: 5,   max: 100,  step: 5,   unit: "pts" },
  PHONE_SCORE_3RD           : { label: "Phone score (3rd+)",         min: 5,   max: 150,  step: 5,   unit: "pts" },
  BOOK_SCORE                : { label: "Book score",                 min: 5,   max: 80,   step: 5,   unit: "pts" },
  HEADPHONE_SCORE           : { label: "Headphone score",            min: 5,   max: 80,   step: 5,   unit: "pts" },
  EARBUD_SCORE              : { label: "Earbud score",               min: 5,   max: 80,   step: 5,   unit: "pts" },
  MULTI_PEOPLE_SCORE_2ND    : { label: "Multiple people score (2nd+)", min: 5, max: 100,  step: 5,   unit: "pts" },
  MULTI_PEOPLE_SCORE_3RD    : { label: "Multiple people score (3rd+)", min: 5, max: 150,  step: 5,   unit: "pts" },
  NO_PERSON_SCORE_1         : { label: "No person score (tier 1)",   min: 5,   max: 100,  step: 5,   unit: "pts" },
  NO_PERSON_SCORE_2         : { label: "No person score (tier 2)",   min: 5,   max: 150,  step: 5,   unit: "pts" },
  NO_PERSON_DUR_1           : { label: "No person tier-1 after",     min: 2,   max: 30,   step: 1,   unit: "s" },
  NO_PERSON_DUR_2           : { label: "No person tier-2 after",     min: 5,   max: 60,   step: 1,   unit: "s" },
  MULTI_PEOPLE_TERMINATE_S  : { label: "Multiple people → terminate",min: 5,   max: 60,   step: 5,   unit: "s" },
  NO_PERSON_TERMINATE_S     : { label: "No person → terminate",      min: 5,   max: 60,   step: 5,   unit: "s" },
  FAKE_PRESENCE_SCORE_1     : { label: "Fake presence score (tier 1)",min: 5,  max: 100,  step: 5,   unit: "pts" },
  FAKE_PRESENCE_SCORE_2     : { label: "Fake presence score (tier 2)",min: 5,  max: 150,  step: 5,   unit: "pts" },
  FAKE_PRESENCE_DUR_1       : { label: "Fake presence tier-1 after", min: 2,   max: 60,   step: 1,   unit: "s" },
  FAKE_PRESENCE_DUR_2       : { label: "Fake presence tier-2 after", min: 5,   max: 120,  step: 5,   unit: "s" },
};

const COOLDOWN_KEYS = [
  "looking_away","looking_down","looking_up","looking_side",
  "speaker_audio","partial_face","face_hidden","fake_presence",
  "phone","multiple_people","no_person","book","headphone","earbud",
];

// ── UI primitives ─────────────────────────────────────────────────────────────

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest mb-3 mt-1" style={{ color: "var(--muted)" }}>
      {children}
    </p>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between gap-4 py-2.5 px-3 rounded-lg cursor-pointer"
      style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
      <span className="text-sm">{label}</span>
      <div
        onClick={() => onChange(!checked)}
        className="relative w-10 h-5 rounded-full transition-colors duration-200 flex-shrink-0"
        style={{ background: checked ? "#22c55e" : "rgba(255,255,255,0.12)", cursor: "pointer" }}
      >
        <div
          className="absolute top-0.5 w-4 h-4 rounded-full transition-transform duration-200"
          style={{ background: "#fff", transform: checked ? "translateX(22px)" : "translateX(2px)" }}
        />
      </div>
    </label>
  );
}

function NumInput({
  label, value, min, max, step, unit, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number;
  unit?: string; onChange: (v: number) => void;
}) {
  const [local, setLocal] = useState(String(value));

  useEffect(() => { setLocal(String(value)); }, [value]);

  function commit(raw: string) {
    const n = parseFloat(raw);
    if (!isNaN(n)) {
      const clamped = Math.min(max, Math.max(min, n));
      onChange(clamped);
      setLocal(String(clamped));
    } else {
      setLocal(String(value));
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <span className="text-sm">{label}</span>
        {unit && <span className="text-xs" style={{ color: "var(--muted)" }}>{unit}</span>}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min} max={max} step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="flex-1 accent-blue-500"
          style={{ accentColor: "#3b82f6" }}
        />
        <input
          type="number"
          min={min} max={max} step={step}
          value={local}
          onChange={e => setLocal(e.target.value)}
          onBlur={e => commit(e.target.value)}
          onKeyDown={e => e.key === "Enter" && commit(local)}
          className="w-20 px-2 py-1 text-sm text-right rounded font-mono"
          style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        />
      </div>
    </div>
  );
}

function CooldownTable({
  score, warn, api, onChange,
}: {
  score: Record<string, number>;
  warn:  Record<string, number>;
  api:   Record<string, number>;
  onChange: (section: "score" | "warn" | "api", key: string, v: number) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-xl" style={{ border: "1px solid var(--border)" }}>
      <table className="w-full text-sm">
        <thead>
          <tr style={{ background: "var(--surface2)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left px-4 py-2.5 font-semibold" style={{ color: "var(--muted)" }}>Event</th>
            <th className="px-4 py-2.5 font-semibold text-right" style={{ color: "#f59e0b" }}>Score (s)</th>
            <th className="px-4 py-2.5 font-semibold text-right" style={{ color: "#f97316" }}>Warn (s)</th>
            <th className="px-4 py-2.5 font-semibold text-right" style={{ color: "#ef4444" }}>Alert (s)</th>
          </tr>
        </thead>
        <tbody>
          {COOLDOWN_KEYS.map((key, i) => (
            <tr key={key}
              style={{ background: i % 2 === 0 ? "var(--surface)" : "var(--surface2)", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
              <td className="px-4 py-2 font-mono text-xs" style={{ color: "var(--muted)" }}>{key}</td>
              {(["score", "warn", "api"] as const).map(col => {
                const map = col === "score" ? score : col === "warn" ? warn : api;
                const v = map[key] ?? 0;
                return (
                  <td key={col} className="px-4 py-2 text-right">
                    <input
                      type="number"
                      min={0} max={120} step={1}
                      value={v}
                      onChange={e => onChange(col, key, parseFloat(e.target.value) || 0)}
                      className="w-16 px-2 py-0.5 text-sm text-right rounded font-mono"
                      style={{ background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)", color: "var(--foreground)" }}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Per-backend settings panel ─────────────────────────────────────────────

function SettingsPanel({ backendUrl }: { backendUrl: string }) {
  const api = createApi(backendUrl);

  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [draft,    setDraft]    = useState<AdminSettings | null>(null);
  const [tab,      setTab]      = useState<"detection" | "thresholds" | "objects" | "scoring" | "cooldowns">("detection");
  const [saving,   setSaving]   = useState(false);
  const [saved,    setSaved]    = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const s = await api.getAdminSettings();
      setSettings(s);
      setDraft(JSON.parse(JSON.stringify(s)));
      setError(null);
    } catch {
      setError("Could not load settings");
    }
  }, [backendUrl]);

  useEffect(() => { load(); }, [load]);

  async function save() {
    if (!draft) return;
    setSaving(true);
    setError(null);
    try {
      await api.saveAdminSettings(draft);
      setSettings(JSON.parse(JSON.stringify(draft)));
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {
      setError("Save failed — check backend");
    } finally {
      setSaving(false);
    }
  }

  function reset() {
    if (settings) setDraft(JSON.parse(JSON.stringify(settings)));
  }

  function setDetect(key: string, v: boolean) {
    setDraft(d => d ? { ...d, detection: { ...d.detection, [key]: v } } : d);
  }
  function setThreshold(key: string, v: number) {
    setDraft(d => d ? { ...d, thresholds: { ...d.thresholds, [key]: v } } : d);
  }
  function setObject(key: string, v: number) {
    setDraft(d => d ? { ...d, objects: { ...d.objects, [key]: v } } : d);
  }
  function setScoring(key: string, v: number) {
    setDraft(d => d ? { ...d, scoring: { ...d.scoring, [key]: v } } : d);
  }
  function setCooldown(section: "score" | "warn" | "api", key: string, v: number) {
    setDraft(d => d ? {
      ...d,
      cooldowns: { ...d.cooldowns, [section]: { ...d.cooldowns[section], [key]: v } },
    } : d);
  }

  const isDirty = JSON.stringify(draft) !== JSON.stringify(settings);

  const TABS: { id: typeof tab; label: string }[] = [
    { id: "detection",  label: "Detections" },
    { id: "thresholds", label: "Thresholds" },
    { id: "objects",    label: "Objects" },
    { id: "scoring",    label: "Scoring" },
    { id: "cooldowns",  label: "Cooldowns" },
  ];

  if (error && !draft) {
    return (
      <div className="rounded-xl p-6 text-center" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <p className="text-sm" style={{ color: "#ef4444" }}>{error}</p>
        <button onClick={load} className="mt-3 text-xs px-3 py-1.5 rounded"
          style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>Retry</button>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="rounded-xl p-6 text-center text-sm" style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)" }}>
        Loading…
      </div>
    );
  }

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      {/* Tab bar */}
      <div className="flex border-b" style={{ borderColor: "var(--border)" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className="px-5 py-3 text-sm font-medium transition-colors"
            style={{
              color: tab === t.id ? "var(--foreground)" : "var(--muted)",
              borderBottom: tab === t.id ? "2px solid #3b82f6" : "2px solid transparent",
              background: "transparent",
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-5">

        {/* ── Detections ── */}
        {tab === "detection" && (
          <div className="space-y-2">
            <SectionHeader>Head / Gaze</SectionHeader>
            {["DETECT_LOOKING_AWAY","DETECT_LOOKING_DOWN","DETECT_LOOKING_UP","DETECT_LOOKING_SIDE",
              "DETECT_FACE_HIDDEN","DETECT_PARTIAL_FACE","DETECT_FAKE_PRESENCE","DETECT_SPEAKER_AUDIO"].map(k => (
              <Toggle key={k} label={DETECTION_LABELS[k] ?? k}
                checked={!!draft.detection[k]} onChange={v => setDetect(k, v)} />
            ))}
            <div className="pt-2">
              <SectionHeader>Objects</SectionHeader>
            </div>
            {["DETECT_PHONE","DETECT_BOOK","DETECT_HEADPHONE","DETECT_EARBUD","DETECT_MULTIPLE_PEOPLE"].map(k => (
              <Toggle key={k} label={DETECTION_LABELS[k] ?? k}
                checked={!!draft.detection[k]} onChange={v => setDetect(k, v)} />
            ))}
            <p className="text-xs pt-1" style={{ color: "var(--muted)" }}>
              Detection toggles apply immediately to all active sessions.
            </p>
          </div>
        )}

        {/* ── Thresholds ── */}
        {tab === "thresholds" && (
          <div className="space-y-5">
            <SectionHeader>Duration Gates</SectionHeader>
            {["LOOKING_AWAY_THRESHOLD","GAZE_THRESHOLD"].map(k => {
              const m = THRESHOLD_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.thresholds[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setThreshold(k, v)} />;
            })}
            <SectionHeader>Head Pose Angles</SectionHeader>
            {["LOOK_AWAY_YAW","LOOK_DOWN_PITCH","LOOK_UP_PITCH","GAZE_LEFT","GAZE_RIGHT"].map(k => {
              const m = THRESHOLD_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.thresholds[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setThreshold(k, v)} />;
            })}
            <SectionHeader>Face Size & Blink</SectionHeader>
            {["EAR_THRESHOLD","MIN_FACE_WIDTH","MIN_FACE_HEIGHT"].map(k => {
              const m = THRESHOLD_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.thresholds[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setThreshold(k, v)} />;
            })}
            <p className="text-xs pt-1" style={{ color: "var(--muted)" }}>
              Threshold changes apply to new sessions only.
            </p>
          </div>
        )}

        {/* ── Objects ── */}
        {tab === "objects" && (
          <div className="space-y-5">
            <SectionHeader>Vote Windows</SectionHeader>
            {["OBJECT_WINDOW","PHONE_MIN_VOTES","BOOK_MIN_VOTES","HEADPHONE_MIN_VOTES","EARBUD_MIN_VOTES","OBJECT_MIN_VOTES"].map(k => {
              const m = OBJECT_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.objects[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setObject(k, v)} />;
            })}
            <SectionHeader>YOLO Confidence Thresholds</SectionHeader>
            {["YOLO_DEFAULT_CONF","YOLO_PHONE_CONF","YOLO_BOOK_CONF","YOLO_AUDIO_CONF"].map(k => {
              const m = OBJECT_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.objects[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setObject(k, v)} />;
            })}
            <p className="text-xs pt-1" style={{ color: "var(--muted)" }}>
              Object detection changes apply to new sessions only.
            </p>
          </div>
        )}

        {/* ── Scoring ── */}
        {tab === "scoring" && (
          <div className="space-y-5">
            <SectionHeader>State Thresholds</SectionHeader>
            {["STATE_WARNING","STATE_HIGH_RISK","STATE_ADMIN","DECAY_AMOUNT"].map(k => {
              const m = SCORING_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.scoring[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setScoring(k, v)} />;
            })}
            <SectionHeader>Tab Switch</SectionHeader>
            {["TAB_SWITCH_SCORE","TAB_SWITCH_TERMINATE_COUNT"].map(k => {
              const m = SCORING_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.scoring[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setScoring(k, v)} />;
            })}
            <SectionHeader>Gaze & Head</SectionHeader>
            {["GAZE_SCORE"].map(k => {
              const m = SCORING_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.scoring[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setScoring(k, v)} />;
            })}
            <SectionHeader>Objects</SectionHeader>
            {["PHONE_SCORE_2ND","PHONE_SCORE_3RD","BOOK_SCORE","HEADPHONE_SCORE","EARBUD_SCORE",
              "MULTI_PEOPLE_SCORE_2ND","MULTI_PEOPLE_SCORE_3RD"].map(k => {
              const m = SCORING_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.scoring[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setScoring(k, v)} />;
            })}
            <SectionHeader>Duration Events</SectionHeader>
            {["NO_PERSON_SCORE_1","NO_PERSON_SCORE_2","NO_PERSON_DUR_1","NO_PERSON_DUR_2",
              "NO_PERSON_TERMINATE_S","MULTI_PEOPLE_TERMINATE_S",
              "FAKE_PRESENCE_SCORE_1","FAKE_PRESENCE_SCORE_2",
              "FAKE_PRESENCE_DUR_1","FAKE_PRESENCE_DUR_2"].map(k => {
              const m = SCORING_META[k];
              return <NumInput key={k} label={m.label} unit={m.unit}
                value={draft.scoring[k] ?? 0} min={m.min} max={m.max} step={m.step}
                onChange={v => setScoring(k, v)} />;
            })}
            <p className="text-xs pt-1" style={{ color: "var(--muted)" }}>
              Scoring changes apply immediately to all active and future sessions.
            </p>
          </div>
        )}

        {/* ── Cooldowns ── */}
        {tab === "cooldowns" && (
          <div className="space-y-4">
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              Score = how often score is added (s). Warn = min gap between soft warnings (s). Alert = min gap between API alerts (s).<br />
              Rule: Alert = Score, Warn ≤ Score.
            </p>
            <CooldownTable
              score={draft.cooldowns.score}
              warn={draft.cooldowns.warn}
              api={draft.cooldowns.api}
              onChange={setCooldown}
            />
            <p className="text-xs pt-1" style={{ color: "var(--muted)" }}>
              Cooldown changes apply immediately.
            </p>
          </div>
        )}
      </div>

      {/* Save bar */}
      <div className="flex items-center gap-3 px-5 py-3 border-t" style={{ borderColor: "var(--border)", background: "var(--surface2)" }}>
        {error && <p className="text-xs flex-1" style={{ color: "#ef4444" }}>{error}</p>}
        {!error && saved && <p className="text-xs flex-1" style={{ color: "#22c55e" }}>Saved successfully</p>}
        {!error && !saved && <p className="text-xs flex-1" style={{ color: "var(--muted)" }}>{isDirty ? "Unsaved changes" : "All changes saved"}</p>}
        <button onClick={reset} disabled={!isDirty || saving}
          className="px-4 py-1.5 text-sm rounded transition-opacity"
          style={{ background: "var(--surface)", border: "1px solid var(--border)", opacity: isDirty ? 1 : 0.4 }}>
          Reset
        </button>
        <button onClick={save} disabled={!isDirty || saving}
          className="px-4 py-1.5 text-sm rounded font-medium transition-opacity"
          style={{ background: isDirty ? "#3b82f6" : "rgba(59,130,246,0.3)", color: "#fff", opacity: isDirty ? 1 : 0.5 }}>
          {saving ? "Saving…" : "Save Changes"}
        </button>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  return (
    <div className="min-h-screen p-6" style={{ background: "var(--background)", color: "var(--foreground)" }}>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link href="/admin" className="text-sm px-3 py-1.5 rounded"
          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)" }}>
          ← Dashboard
        </Link>
        <h1 className="text-xl font-bold">Settings</h1>
      </div>

      {/* One panel per backend */}
      <div className={`grid gap-6 ${BACKEND_URLS.length > 1 ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1 max-w-2xl"}`}>
        {BACKEND_URLS.map(url => (
          <div key={url}>
            {BACKEND_URLS.length > 1 && (
              <p className="text-xs font-semibold mb-2 font-mono" style={{ color: "var(--muted)" }}>
                {portLabel(url)}
              </p>
            )}
            <SettingsPanel backendUrl={url} />
          </div>
        ))}
      </div>
    </div>
  );
}
