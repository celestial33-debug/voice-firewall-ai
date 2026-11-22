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

export default function App() {
  const [mode, setMode] = useState<Mode>("idle");
  const [statusNote, setStatusNote] = useState<string>("Ready — zero audio stored on disk.");
  const [clips, setClips] = useState<string[]>([]);
  const [selectedClip, setSelectedClip] = useState<string>("test_scam.wav");
  const [playAudioAloud, setPlayAudioAloud] = useState<boolean>(true);
  const [latest, setLatest] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);

  // Discover demo_assets/*.wav clips on load
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
      setStatusNote("Backend offline — start uvicorn backend.main:app on port 8000.");
    }
  };

  useEffect(() => {
    refreshClips();
    return () => stopAll();
  }, []);

  const stopAll = () => {
    wsRef.current?.close();
    wsRef.current = null;

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
      setStatusNote(`Replay finished (${msg.clip}) — audio buffer purged from RAM.`);
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
    setHistory((prev) => [telemetry, ...prev].slice(0, 8));
  };

  const startLiveMic = async () => {
    stopAll();
    setLatest(null);
    setHistory([]);
    setMode("live_mic");
    setStatusNote("Listening to live microphone at 16 kHz...");

    const ws = new WebSocket(`${API_WS}/ws/call`);
    ws.onmessage = (e) => handleServerMessage(e.data);
    ws.onclose = () => {
      setMode("idle");
      setStatusNote("Live call ended — RAM buffer purged.");
    };
    wsRef.current = ws;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const ctx = new AudioContext({ sampleRate: 16000 });
    ctxRef.current = ctx;

    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const f32 = event.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(pcm16.buffer);
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  };

  const startSimulatedReplay = () => {
    if (!selectedClip) return;
    stopAll();
    setLatest(null);
    setHistory([]);
    setMode("replay");
    setStatusNote(`Replaying simulated attack: ${selectedClip} (no microphone required)...`);

    const ws = new WebSocket(`${API_WS}/ws/replay/${encodeURIComponent(selectedClip)}`);
    ws.onopen = () => {
      if (playAudioAloud) {
        const audio = new Audio(`${API_HTTP}/api/clips/${encodeURIComponent(selectedClip)}`);
        audioElRef.current = audio;
        audio.play().catch(() => {
          setStatusNote(`Streaming ${selectedClip} (browser blocked autoplay — telemetry still live).`);
        });
      }
    };
    ws.onmessage = (e) => handleServerMessage(e.data);
    ws.onerror = () => {
      setStatusNote("WebSocket error — verify backend is running on 127.0.0.1:8000.");
      stopAll();
    };
    wsRef.current = ws;
  };

  const riskPct = Math.round((latest?.fused_risk ?? 0) * 100);
  const bannerBg =
    latest?.alert_level === "CRITICAL_SCAM"
      ? "#dc2626"
      : latest?.alert_level === "CAUTION"
      ? "#d97706"
      : "#15803d";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "#f8fafc",
        fontFamily: "system-ui, -apple-system, sans-serif",
        padding: "2rem 1rem",
      }}
    >
      <div style={{ maxWidth: 840, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 28 }}>Voice Call Firewall AI</h1>
            <p style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: 14 }}>
              Dual-Signal Acoustic Deepfake + Social-Engineering Protection
            </p>
          </div>
          <span
            style={{
              padding: "6px 12px",
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 600,
              background: "#0f172a",
              border: "1px solid #334155",
              color: "#38bdf8",
            }}
          >
            Zero-Retention RAM Only
          </span>
        </header>

        {/* Control Bar: Live Mic + Bulletproof Stage Demo Replay */}
        <section
          style={{
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 16,
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            alignItems: "center",
          }}
        >
          <button
            onClick={mode === "live_mic" ? stopAll : startLiveMic}
            disabled={mode === "replay"}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              fontWeight: 700,
              cursor: mode === "replay" ? "not-allowed" : "pointer",
              background: mode === "live_mic" ? "#ef4444" : "#2563eb",
              color: "#fff",
              opacity: mode === "replay" ? 0.5 : 1,
            }}
          >
            {mode === "live_mic" ? "Stop Mic & Purge RAM" : "Start Live Mic Guard"}
          </button>

          <div style={{ height: 28, width: 1, background: "#334155" }} />

          <select
            value={selectedClip}
            onChange={(e) => setSelectedClip(e.target.value)}
            disabled={mode !== "idle"}
            style={{
              padding: "9px 12px",
              borderRadius: 8,
              background: "#1e293b",
              color: "#f8fafc",
              border: "1px solid #475569",
              fontSize: 14,
            }}
          >
            {clips.length === 0 ? (
              <option value="">No .wav clips in demo_assets/</option>
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
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              fontWeight: 700,
              cursor: mode === "live_mic" ? "not-allowed" : "pointer",
              background: mode === "replay" ? "#ef4444" : "#e11d48",
              color: "#fff",
              opacity: mode === "live_mic" ? 0.5 : 1,
            }}
          >
            {mode === "replay" ? "Stop Replay" : "Replay Simulated Attack"}
          </button>

          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={playAudioAloud}
              onChange={(e) => setPlayAudioAloud(e.target.checked)}
              disabled={mode !== "idle"}
            />
            Play audio aloud
          </label>
        </section>

        <p style={{ fontSize: 13, color: "#94a3b8", margin: "10px 4px 20px" }}>{statusNote}</p>

        {/* Live Threat Banner */}
        {latest && (
          <>
            <div
              style={{
                padding: "18px 20px",
                borderRadius: 12,
                background: bannerBg,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 12,
                marginBottom: 16,
              }}
            >
              <div>
                <div style={{ fontSize: 12, opacity: 0.9, letterSpacing: 1 }}>FIREWALL VERDICT</div>
                <div style={{ fontSize: 24, fontWeight: 800 }}>
                  {latest.alert_level.replace("_", " ")} — {riskPct}% Risk
                </div>
              </div>
              <div style={{ textAlign: "right", fontSize: 13 }}>
                <div>Window #{latest.window_index}</div>
                <div>Latency: <strong>{latest.latency_ms} ms</strong></div>
                {latest.synergy_triggered && (
                  <div style={{ marginTop: 4, fontWeight: 700 }}>DUAL-SIGNAL SYNERGY BOOST</div>
                )}
              </div>
            </div>

            {/* Dual-Signal Score Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 16 }}>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>Acoustic Deepfake (Wav2Vec2)</div>
                <div style={{ fontSize: 34, fontWeight: 800, marginTop: 4 }}>
                  {Math.round(latest.acoustic_prob * 100)}%
                </div>
                <div style={{ height: 8, background: "#1e293b", borderRadius: 4, marginTop: 10, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${Math.round(latest.acoustic_prob * 100)}%`,
                      height: "100%",
                      background: latest.acoustic_prob >= 0.45 ? "#f43f5e" : "#38bdf8",
                      transition: "width 250ms ease",
                    }}
                  />
                </div>
              </div>

              <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 16 }}>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>Social Engineering (Whisper + DeBERTa)</div>
                <div style={{ fontSize: 34, fontWeight: 800, marginTop: 4 }}>
                  {Math.round(latest.coercion_prob * 100)}%
                </div>
                <div style={{ height: 8, background: "#1e293b", borderRadius: 4, marginTop: 10, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${Math.round(latest.coercion_prob * 100)}%`,
                      height: "100%",
                      background: latest.coercion_prob >= 0.55 ? "#f59e0b" : "#22c55e",
                      transition: "width 250ms ease",
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Evidence Feed */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 16 }}>
              <h3 style={{ margin: "0 0 12px", fontSize: 15, color: "#cbd5e1" }}>
                Explainable Evidence Stream
              </h3>
              {history.map((item) => (
                <div
                  key={item.window_index}
                  style={{
                    padding: "10px 12px",
                    marginBottom: 8,
                    borderRadius: 8,
                    background: "#1e293b",
                    borderLeft: `4px solid ${
                      item.alert_level === "CRITICAL_SCAM"
                        ? "#ef4444"
                        : item.alert_level === "CAUTION"
                        ? "#f59e0b"
                        : "#22c55e"
                    }`,
                  }}
                >
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>
                    Window #{item.window_index} • Risk {Math.round(item.fused_risk * 100)}% • {item.latency_ms} ms
                  </div>
                  <div style={{ fontSize: 15, marginBottom: item.triggers.length ? 6 : 0 }}>
                    "{item.transcript || "..."}"
                  </div>
                  {item.triggers.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {item.triggers.map((t, i) => (
                        <span
                          key={i}
                          style={{
                            fontSize: 11,
                            padding: "2px 8px",
                            borderRadius: 999,
                            background: "rgba(239,68,68,0.2)",
                            color: "#fca5a5",
                            border: "1px solid rgba(239,68,68,0.4)",
                          }}
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

