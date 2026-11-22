# test_ws_client.py
import asyncio
import json
import sys
import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample

WS_URL = "ws://127.0.0.1:8000/ws/call"
SAMPLE_RATE = 16000
CHUNK_SEC = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


def load_as_pcm16_int16(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        target_samples = int(len(audio) * (SAMPLE_RATE / sr))
        audio = resample(audio, target_samples).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


async def stream_audio(audio_path: str) -> None:
    pcm16 = load_as_pcm16_int16(audio_path)
    total_sec = len(pcm16) / SAMPLE_RATE
    print(f"[Client] Loaded {audio_path} ({total_sec:.2f}s). Connecting to {WS_URL}...")

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("[Client] Connected! Streaming 0.25s live audio chunks...\n")

        async def receiver() -> None:
            try:
                async for message in ws:
                    data = json.loads(message)
                    print("=== LIVE ALERT RECEIVED ===")
                    print(
                        f"  Window #{data['window_index']} | "
                        f"STATUS: {data['alert_level']} "
                        f"(Fused Risk: {data['fused_risk'] * 100:.1f}%) | "
                        f"Latency: {data['latency_ms']} ms"
                    )
                    print(
                        f"  Acoustic: {data['acoustic_prob']:.3f} | "
                        f"Coercion: {data['coercion_prob']:.3f} | "
                        f"Synergy: {data['synergy_triggered']}"
                    )
                    print(f"  Transcript: \"{data['transcript']}\"")
                    if data["triggers"]:
                        print(f"  Triggers:   {', '.join(data['triggers'])}")
                    print()
            except websockets.exceptions.ConnectionClosed:
                pass

        async def sender() -> None:
            for start in range(0, len(pcm16), CHUNK_SAMPLES):
                chunk = pcm16[start : start + CHUNK_SAMPLES]
                await ws.send(chunk.tobytes())
                await asyncio.sleep(CHUNK_SEC)

            print("[Client] Audio stream finished. Waiting for final inference results...")
            await asyncio.sleep(4.0)
            await ws.close()

        await asyncio.gather(sender(), receiver())
        print("[Client] Test complete.")


if __name__ == "__main__":
    clip = sys.argv[1] if len(sys.argv) > 1 else "demo_assets/test_scam.wav"
    asyncio.run(stream_audio(clip))
