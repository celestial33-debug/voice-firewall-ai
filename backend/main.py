# backend/main.py
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from scipy.signal import resample

from backend.models.fusion import AudioRingBuffer, RiskFusionEngine
from backend.models.pipeline import FirewallModels

DEMO_DIR = Path("demo_assets").resolve()
SAMPLE_RATE = 16000
CHUNK_SEC = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)

models: FirewallModels | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = FirewallModels()
    yield
    models = None


app = FastAPI(title="Voice Call Firewall API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_clip(filename: str) -> Path:
    safe_name = Path(filename).name
    target = (DEMO_DIR / safe_name).resolve()
    if target.parent != DEMO_DIR or not target.is_file() or target.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail=f"Demo clip '{safe_name}' not found.")
    return target


def _load_wav_as_pcm16(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        target_samples = int(len(audio) * (SAMPLE_RATE / sr))
        audio = resample(audio, target_samples).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


async def _evaluate_window(
    window: np.ndarray, window_index: int, fusion: RiskFusionEngine
) -> dict | None:
    t0 = time.perf_counter()

    has_speech = await asyncio.to_thread(models.has_speech, window)
    if not has_speech:
        print(f"[Pipeline] Window #{window_index} skipped (silence).")
        return None

    acoustic_res, coercion_res = await asyncio.gather(
        asyncio.to_thread(models.score_acoustic, window),
        asyncio.to_thread(models.score_coercion, window),
    )

    acoustic_prob = acoustic_res["acoustic_prob"]
    coercion_prob = coercion_res["coercion_prob"]
    verdict = fusion.update(acoustic_prob, coercion_prob)
    latency_ms = round((time.perf_counter() - t0) * 1000)

    payload = {
        "event": "telemetry",
        "window_index": window_index,
        "acoustic_prob": acoustic_prob,
        "coercion_prob": coercion_prob,
        "instant_risk": verdict["instant_risk"],
        "fused_risk": verdict["fused_risk"],
        "alert_level": verdict["alert_level"],
        "synergy_triggered": verdict["synergy_triggered"],
        "transcript": coercion_res["transcript"],
        "triggers": coercion_res["triggers"],
        "latency_ms": latency_ms,
    }
    print(
        f"[Pipeline] Win #{window_index} | {payload['alert_level']} "
        f"(Fused={payload['fused_risk']:.3f}) | {latency_ms} ms"
    )
    return payload


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": models is not None,
        "cuda_available": bool(models and models.has_cuda),
    }


@app.get("/api/clips")
async def list_demo_clips():
    if not DEMO_DIR.exists():
        return {"clips": []}
    clips = sorted(
        p.name
        for p in DEMO_DIR.glob("*.wav")
        if not p.name.endswith("_raw.wav")
    )
    return {"clips": clips}


@app.get("/api/clips/{filename}")
async def get_demo_clip(filename: str):
    clip_path = _resolve_clip(filename)
    return FileResponse(path=str(clip_path), media_type="audio/wav", filename=clip_path.name)


@app.websocket("/ws/call")
async def call_stream(ws: WebSocket):
    await ws.accept()
    print("[WS/Call] Client connected — live microphone monitoring started.")

    ring = AudioRingBuffer(sample_rate=SAMPLE_RATE, window_sec=4.0, hop_sec=2.0)
    fusion = RiskFusionEngine(alpha=0.55)
    window_index = 0

    try:
        while True:
            raw_pcm = await ws.receive_bytes()
            window = ring.append_pcm16(raw_pcm)
            if window is None:
                continue

            window_index += 1
            payload = await _evaluate_window(window, window_index, fusion)
            if payload is not None:
                await ws.send_json(payload)
    except WebSocketDisconnect:
        print("[WS/Call] Client disconnected cleanly.")
    finally:
        ring.purge()
        print("[WS/Call] Audio ring buffer purged from RAM.")


@app.websocket("/ws/replay/{filename}")
async def replay_attack_stream(ws: WebSocket, filename: str):
    await ws.accept()
    try:
        clip_path = _resolve_clip(filename)
    except HTTPException as exc:
        await ws.send_json({"event": "error", "detail": exc.detail})
        await ws.close()
        return

    print(f"[WS/Replay] Starting simulated attack replay: {clip_path.name}")
    pcm16 = await asyncio.to_thread(_load_wav_as_pcm16, clip_path)

    ring = AudioRingBuffer(sample_rate=SAMPLE_RATE, window_sec=4.0, hop_sec=2.0)
    fusion = RiskFusionEngine(alpha=0.55)
    window_queue: asyncio.Queue[tuple[int, np.ndarray] | None] = asyncio.Queue()

    async def producer() -> None:
        window_index = 0
        start_time = time.perf_counter()
        step = 0

        for pos in range(0, len(pcm16), CHUNK_SAMPLES):
            chunk = pcm16[pos : pos + CHUNK_SAMPLES]
            window = ring.append_pcm16(chunk.tobytes())
            if window is not None:
                window_index += 1
                await window_queue.put((window_index, window))

            step += 1
            next_tick = start_time + (step * CHUNK_SEC)
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        # Flush any trailing audio if the clip is shorter than 4s or ended mid-hop
        if window_index == 0 and len(ring.buffer) > 0:
            padded = np.pad(ring.buffer, (0, max(0, ring.window_samples - len(ring.buffer))))
            await window_queue.put((1, padded))

        await window_queue.put(None)

    async def consumer() -> None:
        while True:
            item = await window_queue.get()
            if item is None:
                break
            win_idx, window = item
            payload = await _evaluate_window(window, win_idx, fusion)
            if payload is not None:
                await ws.send_json(payload)

    try:
        await asyncio.gather(producer(), consumer())
        await ws.send_json({"event": "replay_complete", "clip": clip_path.name})
        print(f"[WS/Replay] Replay finished: {clip_path.name}")
    except WebSocketDisconnect:
        print("[WS/Replay] Client stopped replay early.")
    finally:
        ring.purge()
        print("[WS/Replay] Replay audio ring buffer purged from RAM.")

from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path("frontend/dist").resolve()
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
