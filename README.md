---
title: Voice Call Firewall AI
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# Voice Call Firewall AI

Real-time, privacy-first audio firewall that detects **AI voice cloning** and **live social-engineering coercion** simultaneously during phone calls—running locally on CPU in ~2.2 seconds with zero audio stored to disk.

## System Architecture
- **Silero VAD (ONNX):** Strips silence in real time.
- **Branch A — Acoustic Forensics:** `wav2vec2-deepfake-voice-detector` flags neural TTS artifacts.
- **Branch B — Coercion Analyzer:** `Faster-Whisper` + zero-shot NLI (`xtremedistil` on CPU / `deberta-v3` on GPU) + OTP/remote-access guardrails.
- **EMA Risk Fusion:** Combines both branches over a 4-second sliding window (2-second hop) with a dual-signal synergy boost and RAM zero-fill purge on hangup.

## Key Engineering Highlights
- **8.9x CPU Speedup:** Cut pure-CPU latency from **19,900 ms** down to **2,235 ms** via multi-core thread partitioning and quantized `int8` inference.
- **8 kHz Telephony Resilience:** Benchmarked against 8 kHz G.711 u-law phone compression (`evaluation/benchmark.py`).
- **Zero-Retention Privacy:** Raw PCM stays strictly in a volatile RAM ring buffer and is purged on disconnect.

## Run Locally

Terminal 1 (Backend):

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch torchaudio -r requirements.txt
    uvicorn backend.main:app --host 127.0.0.1 --port 8000

Terminal 2 (Frontend):

    cd frontend
    npm install
    npm run dev