# test_pipeline.py
import sys
import time
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample
from backend.models.pipeline import FirewallModels


def load_wav_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        target_samples = int(len(audio) * (16000 / sr))
        audio = resample(audio, target_samples).astype(np.float32)
    return audio


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "demo_assets/test_scam.wav"

    t_load = time.perf_counter()
    models = FirewallModels()
    print(f"\nModel load time: {time.perf_counter() - t_load:.2f} s")

    print(f"\n--- Testing audio file: {audio_path} ---")
    audio = load_wav_16k(audio_path)
    print(f"Duration: {len(audio) / 16000:.2f} s | Shape: {audio.shape} | Dtype: {audio.dtype}")

    # Stage 1: VAD
    t0 = time.perf_counter()
    speech_present = models.has_speech(audio)
    vad_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not speech_present:
        print(f"VAD detected NO speech ({vad_ms} ms). Check that your audio file is not silent.")
        return

    # Stage 2: Acoustic Deepfake Forensics
    t1 = time.perf_counter()
    acoustic = models.score_acoustic(audio)
    acoustic_ms = round((time.perf_counter() - t1) * 1000, 1)

    # Stage 3: ASR + Social-Engineering Intent
    t2 = time.perf_counter()
    coercion = models.score_coercion(audio)
    coercion_ms = round((time.perf_counter() - t2) * 1000, 1)

    report = {
        "speech_detected": speech_present,
        "acoustic_prob": acoustic["acoustic_prob"],
        "coercion_prob": coercion["coercion_prob"],
        "transcript": coercion["transcript"],
        "triggers": coercion["triggers"],
        "timings_ms": {
            "vad": vad_ms,
            "acoustic_wav2vec2": acoustic_ms,
            "asr_plus_nli": coercion_ms,
            "sequential_total": round(vad_ms + acoustic_ms + coercion_ms, 1),
        },
    }

    print("\n=== PIPELINE OUTPUT ===")
    print(json.dumps(report, indent=2))

    # Sanity check: verify a benign sentence scores low on coercion
    benign_text = "Hey, are we still meeting at the library after class to work on our slides?"
    benign_out = models.nli_pipe(
        benign_text,
        [
            "urgent demand to transfer money or pay immediately",
            "requesting one-time password, PIN, or remote screen access",
        ],
        multi_label=True,
    )
    print(f"\nBenign text max coercion score (should be low): {max(benign_out['scores']):.3f}")


if __name__ == "__main__":
    main()
