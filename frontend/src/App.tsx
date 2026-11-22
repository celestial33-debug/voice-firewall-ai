// frontend/src/App.tsx
import { useEffect, useRef, useState } from "react";

const isLocalDev = window.location.port.startsWith("517");
const API_HTTP = isLocalDev ? "http://127.0.0.1:8000" : window.location.origin;
const API_WS = isLocalDev
  ? "ws://127.0.0.1:8000"
  : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;

interface Telemetry {
  event?: string;
  window_index: number;
  acoustic_prob: number;
  coercion_prob: number;
  instant_risk: number;
  fused_risk: number;
  alert_level: "SAFE" | "CAUTION" | "CRITICAL_SCAM";
  synergy_triggered: boolean;
  transcript: string;
  triggers: string[];
  latency_ms: number;
}

type Mode = "idle" | "live_mic" | "replay";

const ALERT_THEME = {
  SAFE: {
    label: "Call Looks Safe",
    badge: "SAFE",
    accent: "#10b981",
    softBg: "rgba(16, 185, 129, 0.10)",
    border: "rgba(16, 185, 129, 0.28)",
    glow: "0 0 40px rgba(16, 185, 129, 0.12)",
  },
  CAUTION: {
    label: "Suspicious Signals",
    badge: "CAUTION",
    accent: "#f59e0b",
    softBg: "rgba(245, 158, 11, 0.11)",
    border: "rgba(245, 158, 11, 0.32)",
    glow: "0 0 40px rgba(245, 158, 11, 0.14)",
  },
  CRITICAL_SCAM: {
    label: "Scam Call Detected",
    badge: "CRITICAL SCAM",
    accent: "#ef4444",
    softBg: "rgba(239, 68, 68, 0.12)",
    border: "rgba(239, 68, 68, 0.38)",
    glow: "0 0 48px rgba(239, 68, 68, 0.20)",
  },
} as const;

export default function App() {
  const [mode, setMode] = useState<Mode>("idle");
  const [statusNote, setStatusNote] = useState<string>("Ready \u2014 audio stays in RAM only.");
  const [clips, setClips] = useState<string[]>([]);
  const [selectedClip, setSelectedClip] = useState<string>("test_scam.wav");
  const [playAudioAloud, setPlayAudioAloud] = useState<boolean>(true);
  const [latest, setLatest] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);

  const refreshClips = async () => {
    try {
      const res = await fetch(`${API_HTTP}/api/clips`);
      const json = await res.json();
      const list: string[] = json.clips ?? [];
      setClips(list);
      if (list.includes("cloned_attack.wav")) {
        setSelectedClip("cloned_attack.wav");
      } else if (list.length > 0) {
        setSelectedClip(list[0]);
      }
    } catch {
      setStatusNote("Backend offline \u2014 start uvicorn backend.main:app on port 8000.");
    }
  };

  useEffect(() => {
    refreshClips();
    return () => stopAll();
  }, []);

  const stopAll = () => {
    wsRef.current?.close();
    wsRef.current = null;

    processorRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;

    ctxRef.current?.close();
    ctxRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (audioElRef.current) {
      audioElRef.current.pause();
      audioElRef.current.currentTime = 0;
      audioElRef.current = null;
    }

    setMode("idle");
  };

  const handleServerMessage = (raw: string) => {
    const msg = JSON.parse(raw);
    if (msg.event === "replay_complete") {
      setStatusNote(`Finished ${msg.clip} \u2014 RAM buffer purged.`);
      setMode("idle");
      return;
    }
    if (msg.event === "error") {
      setStatusNote(`Error: ${msg.detail}`);
      stopAll();
      return;
    }
    const telemetry = msg as Telemetry;
    setLatest(telemetry);
    setHistory((prev) => [telemetry, ...prev].slice(0, 6));
  };

  const startLiveMic = async () => {
    stopAll();
    setLatest(null);
    setHistory([]);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      setMode("live_mic");
      setStatusNote("Connecting live mic to firewall...");

      const ws = new WebSocket(`${API_WS}/ws/call`);
      ws.onopen = () => setStatusNote("Mic connected \u2014 speak for 5+ seconds...");
      ws.onmessage = (e) => handleServerMessage(e.data);
      ws.onerror = () => {
        setStatusNote("WebSocket /ws/call error \u2014 check uvicorn terminal.");
        stopAll();
      };
      ws.onclose = () => {
        setMode("idle");
        setStatusNote("Live session ended \u2014 RAM buffer purged.");
      };
      wsRef.current = ws;

      const ctx = new AudioContext({ sampleRate: 16000 });
      await ctx.resume();
      ctxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0;

      sourceRef.current = source;
      processorRef.current = processor;

      let chunksSent = 0;
      processor.onaudioprocess = (event) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const f32 = event.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(f32.length);
        let peak = 0;
        for (let i = 0; i < f32.length; i++) {
          const s = Math.max(-1, Math.min(1, f32[i]));
          if (Math.abs(s) > peak) peak = Math.abs(s);
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(pcm16.buffer);
        chunksSent += 1;
        if (chunksSent % 4 === 0) {
          const secs = ((chunksSent * 4096) / ctx.sampleRate).toFixed(1);
          const lvl = Math.round(peak * 100);
          setStatusNote(`Mic live (${secs}s sent) \u2014 input level: ${lvl}%`);
        }
      };

      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(ctx.destination);
    } catch (err) {
      stopAll();
      const msg = err instanceof Error ? err.message : "Permission denied or no mic";
      setStatusNote(`Microphone error: ${msg}`);
    }
  };

  const startSimulatedReplay = () => {
    if (!selectedClip) return;
    stopAll();
    setLatest(null);
    setHistory([]);
    setMode("replay");
    setStatusNote(`Analyzing ${selectedClip}...`);

    const ws = new WebSocket(`${API_WS}/ws/replay/${encodeURIComponent(selectedClip)}`);
    ws.onopen = () => {
      if (playAudioAloud) {
        const audio = new Audio(`${API_HTTP}/api/clips/${encodeURIComponent(selectedClip)}`);
        audioElRef.current = audio;
        audio.play().catch(() => {
          setStatusNote(`Analyzing ${selectedClip} (browser muted autoplay).`);
        });
      }
    };
    ws.onmessage = (e) => handleServerMessage(e.data);
    ws.onerror = () => {
      setStatusNote("Connection error \u2014 verify backend is running on port 8000.");
      stopAll();
    };
    wsRef.current = ws;
  };

  const riskPct = Math.round((latest?.fused_risk ?? 0) * 100);
  const acousticPct = Math.round((latest?.acoustic_prob ?? 0) * 100);
  const coercionPct = Math.round((latest?.coercion_prob ?? 0) * 100);
  const theme = ALERT_THEME[latest?.alert_level ?? "SAFE"];

  // Circular gauge geometry
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeOffset = circumference * (1 - riskPct / 100);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "radial-gradient(circle at 50% -10%, #172033 0%, #090c14 55%, #06080d 100%)",
        color: "#f1f5f9",
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        padding: "2.25rem 1.25rem 3rem",
        boxSizing: "border-box",
      }}
    >
      <style>{`
        @keyframes pulseDot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.45; transform: scale(0.85); }
        }
        button:hover:not(:disabled) { filter: brightness(1.08); }
        select:focus, button:focus { outline: 2px solid rgba(56, 189, 248, 0.5); outline-offset: 2px; }
      `}</style>

      <div style={{ maxWidth: 780, margin: "0 auto" }}>
        {/* Header */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
            marginBottom: 24,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 11,
                background: "linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 6px 20px rgba(37, 99, 235, 0.35)",
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
                Voice Call Firewall
              </h1>
              <p style={{ margin: "2px 0 0", color: "#94a3b8", fontSize: 13 }}>
                Real-time AI voice clone &amp; scam protection
              </p>
            </div>
          </div>

          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              padding: "6px 12px",
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 500,
              background: "rgba(15, 23, 42, 0.85)",
              border: "1px solid rgba(148, 163, 184, 0.18)",
              color: "#cbd5e1",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: "#10b981",
                display: "inline-block",
              }}
            />
            Zero-Retention RAM Only
          </div>
        </header>

        {/* Compact Control Bar */}
        <section
          style={{
            background: "rgba(15, 23, 42, 0.78)",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(148, 163, 184, 0.14)",
            borderRadius: 16,
            padding: "14px 16px",
            display: "flex",
            flexWrap: "wrap",
            gap: 10,
            alignItems: "center",
          }}
        >
          <button
            onClick={mode === "live_mic" ? stopAll : startLiveMic}
            disabled={mode === "replay"}
            style={{
              padding: "10px 16px",
              borderRadius: 10,
              border: "none",
              fontSize: 13,
              fontWeight: 600,
              cursor: mode === "replay" ? "not-allowed" : "pointer",
              background: mode === "live_mic" ? "#ef4444" : "#2563eb",
              color: "#fff",
              opacity: mode === "replay" ? 0.45 : 1,
              transition: "all 150ms ease",
            }}
          >
            {mode === "live_mic" ? "Stop Mic" : "Live Mic"}
          </button>

          <div style={{ height: 24, width: 1, background: "rgba(148, 163, 184, 0.2)" }} />

          <select
            value={selectedClip}
            onChange={(e) => setSelectedClip(e.target.value)}
            disabled={mode !== "idle"}
            style={{
              flex: "1 1 180px",
              padding: "9px 12px",
              borderRadius: 10,
              background: "#0b1220",
              color: "#f1f5f9",
              border: "1px solid rgba(148, 163, 184, 0.22)",
              fontSize: 13,
              cursor: mode !== "idle" ? "not-allowed" : "pointer",
            }}
          >
            {clips.length === 0 ? (
              <option value="">No audio clips found</option>
            ) : (
              clips.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))
            )}
          </select>

          <button
            onClick={mode === "replay" ? stopAll : startSimulatedReplay}
            disabled={mode === "live_mic" || clips.length === 0}
            style={{
              padding: "10px 18px",
              borderRadius: 10,
              border: "none",
              fontSize: 13,
              fontWeight: 600,
              cursor: mode === "live_mic" || clips.length === 0 ? "not-allowed" : "pointer",
              background: mode === "replay" ? "#ef4444" : "#f43f5e",
              color: "#fff",
              opacity: mode === "live_mic" || clips.length === 0 ? 0.45 : 1,
              transition: "all 150ms ease",
            }}
          >
            {mode === "replay" ? "Stop" : "Run Demo Clip"}
          </button>

          <label
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              color: "#94a3b8",
              marginLeft: "auto",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={playAudioAloud}
              onChange={(e) => setPlayAudioAloud(e.target.checked)}
              disabled={mode !== "idle"}
              style={{ accentColor: "#38bdf8" }}
            />
            Audio On
          </label>
        </section>

        {/* Quiet Status Line */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12,
            color: "#94a3b8",
            margin: "12px 6px 20px",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: mode !== "idle" ? "#38bdf8" : "#475569",
              animation: mode !== "idle" ? "pulseDot 1.2s infinite" : "none",
            }}
          />
          <span>{statusNote}</span>
        </div>

        {/* Idle Empty State */}
        {!latest && (
          <div
            style={{
              background: "rgba(15, 23, 42, 0.5)",
              border: "1px dashed rgba(148, 163, 184, 0.2)",
              borderRadius: 18,
              padding: "44px 24px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 16, fontWeight: 600, color: "#e2e8f0", marginBottom: 6 }}>
              Firewall Standing By
            </div>
            <p style={{ margin: 0, fontSize: 13, color: "#94a3b8", maxWidth: 420, marginInline: "auto" }}>
              Select a clip above and click <strong>Run Demo Clip</strong>, or start <strong>Live Mic</strong> to monitor a call in real time.
            </p>
          </div>
        )}

        {/* Active Unified Dashboard */}
        {latest && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Unified Verdict + Signals Hero Card */}
            <section
              style={{
                background: "rgba(15, 23, 42, 0.85)",
                border: `1px solid ${theme.border}`,
                boxShadow: theme.glow,
                borderRadius: 20,
                padding: "22px 24px",
              }}
            >
              {/* Top Row: Circular Gauge + Verdict + Telemetry */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: 20,
                  paddingBottom: 20,
                  borderBottom: "1px solid rgba(148, 163, 184, 0.12)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
                  {/* SVG Risk Gauge */}
                  <div style={{ position: "relative", width: 96, height: 96, flexShrink: 0 }}>
                    <svg width="96" height="96" style={{ transform: "rotate(-90deg)" }}>
                      <circle
                        cx="48"
                        cy="48"
                        r={radius}
                        fill="none"
                        stroke="rgba(148, 163, 184, 0.14)"
                        strokeWidth="8"
                      />
                      <circle
                        cx="48"
                        cy="48"
                        r={radius}
                        fill="none"
                        stroke={theme.accent}
                        strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeOffset}
                        style={{ transition: "stroke-dashoffset 350ms ease, stroke 250ms ease" }}
                      />
                    </svg>
                    <div
                      style={{
                        position: "absolute",
                        inset: 0,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <span style={{ fontSize: 22, fontWeight: 800, lineHeight: 1 }}>{riskPct}%</span>
                      <span style={{ fontSize: 10, color: "#94a3b8", marginTop: 3 }}>RISK</span>
                    </div>
                  </div>

                  {/* Verdict Title */}
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 700,
                          letterSpacing: "0.06em",
                          padding: "3px 9px",
                          borderRadius: 999,
                          background: theme.softBg,
                          color: theme.accent,
                          border: `1px solid ${theme.border}`,
                        }}
                      >
                        {theme.badge}
                      </span>
                      {latest.synergy_triggered && (
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 600,
                            padding: "3px 9px",
                            borderRadius: 999,
                            background: "rgba(244, 63, 94, 0.15)",
                            color: "#fda4af",
                            border: "1px solid rgba(244, 63, 94, 0.3)",
                          }}
                        >
                          Dual-Signal Boost
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
                      {theme.label}
                    </div>
                  </div>
                </div>

                {/* Right Telemetry */}
                <div style={{ textAlign: "right", fontSize: 12, color: "#94a3b8", lineHeight: 1.7 }}>
                  <div>
                    Window <strong style={{ color: "#e2e8f0" }}>#{latest.window_index}</strong>
                  </div>
                  <div>
                    Latency <strong style={{ color: "#e2e8f0" }}>{latest.latency_ms} ms</strong>
                  </div>
                </div>
              </div>

              {/* Bottom Row: Two Clean Signal Bars */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
                  gap: 20,
                  paddingTop: 18,
                }}
              >
                {/* Signal 1: AI Voice Clone */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                    <div>
                      <span style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0" }}>AI Voice Clone</span>
                      <span style={{ fontSize: 11, color: "#64748b", marginLeft: 6 }}>Wav2Vec2</span>
                    </div>
                    <span style={{ fontSize: 18, fontWeight: 700, color: acousticPct >= 45 ? "#fb7185" : "#38bdf8" }}>
                      {acousticPct}%
                    </span>
                  </div>
                  <div style={{ height: 7, background: "rgba(148, 163, 184, 0.12)", borderRadius: 999, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${acousticPct}%`,
                        height: "100%",
                        borderRadius: 999,
                        background: acousticPct >= 45 ? "#f43f5e" : "#38bdf8",
                        transition: "width 300ms ease",
                      }}
                    />
                  </div>
                </div>

                {/* Signal 2: Scam Tactics */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                    <div>
                      <span style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0" }}>Scam Tactics</span>
                      <span style={{ fontSize: 11, color: "#64748b", marginLeft: 6 }}>Whisper + NLI</span>
                    </div>
                    <span style={{ fontSize: 18, fontWeight: 700, color: coercionPct >= 55 ? "#fbbf24" : "#34d399" }}>
                      {coercionPct}%
                    </span>
                  </div>
                  <div style={{ height: 7, background: "rgba(148, 163, 184, 0.12)", borderRadius: 999, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${coercionPct}%`,
                        height: "100%",
                        borderRadius: 999,
                        background: coercionPct >= 55 ? "#f59e0b" : "#10b981",
                        transition: "width 300ms ease",
                      }}
                    />
                  </div>
                </div>
              </div>
            </section>

            {/* Clean Live Transcript & Evidence */}
            <section
              style={{
                background: "rgba(15, 23, 42, 0.72)",
                border: "1px solid rgba(148, 163, 184, 0.12)",
                borderRadius: 18,
                padding: "18px 20px",
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", color: "#94a3b8", marginBottom: 14 }}>
                LIVE TRANSCRIPT &amp; EVIDENCE
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {history.map((item, idx) => {
                  const rowTheme = ALERT_THEME[item.alert_level];
                  const isNewest = idx === 0;
                  return (
                    <div
                      key={item.window_index}
                      style={{
                        padding: isNewest ? "14px 16px" : "10px 14px",
                        borderRadius: 12,
                        background: isNewest ? "rgba(30, 41, 59, 0.75)" : "rgba(15, 23, 42, 0.45)",
                        border: isNewest
                          ? `1px solid ${rowTheme.border}`
                          : "1px solid rgba(148, 163, 184, 0.08)",
                        opacity: isNewest ? 1 : 0.75,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          fontSize: 11,
                          color: "#94a3b8",
                          marginBottom: 6,
                        }}
                      >
                        <span>
                          Window #{item.window_index} {"\u2022"} {item.latency_ms} ms
                        </span>
                        <span style={{ color: rowTheme.accent, fontWeight: 600 }}>
                          {Math.round(item.fused_risk * 100)}% risk
                        </span>
                      </div>

                      <div
                        style={{
                          fontSize: isNewest ? 15 : 13,
                          color: isNewest ? "#f8fafc" : "#cbd5e1",
                          lineHeight: 1.5,
                          marginBottom: item.triggers.length ? 8 : 0,
                        }}
                      >
                        &ldquo;{item.transcript || "Listening..."}&rdquo;
                      </div>

                      {item.triggers.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                          {item.triggers.map((t, i) => (
                            <span
                              key={i}
                              style={{
                                fontSize: 11,
                                padding: "3px 9px",
                                borderRadius: 999,
                                background: "rgba(239, 68, 68, 0.14)",
                                color: "#fca5a5",
                                border: "1px solid rgba(239, 68, 68, 0.28)",
                              }}
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}