# evaluation/benchmark.py
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.models.fusion import RiskFusionEngine
from backend.models.pipeline import FirewallModels

DEMO_DIR = ROOT_DIR / "demo_assets"
OUT_FILE = ROOT_DIR / "evaluation" / "results" / "benchmark_summary.json"
SR = 16000
WIN_SAMPLES = SR * 4
HOP_SAMPLES = SR * 2


def load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = resample(audio, int(len(audio) * (SR / sr))).astype(np.float32)
    return audio


def degrade_to_8khz_mulaw(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(dst)],
        check=True,
    )


def evaluate_clip(models: FirewallModels, audio: np.ndarray) -> dict:
    fusion = RiskFusionEngine(alpha=0.55)
    latencies_ms = []
    max_acoustic = 0.0
    max_coercion = 0.0
    peak_risk = 0.0
    final_level = "SAFE"

    if len(audio) < WIN_SAMPLES:
        windows = [np.pad(audio, (0, WIN_SAMPLES - len(audio)))]
    else:
        windows = [
            audio[start : start + WIN_SAMPLES]
            for start in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES)
        ]

    for win in windows:
        t0 = time.perf_counter()
        if not models.has_speech(win):
            continue
        a_res = models.score_acoustic(win)
        c_res = models.score_coercion(win)
        verdict = fusion.update(a_res["acoustic_prob"], c_res["coercion_prob"])
        latencies_ms.append(round((time.perf_counter() - t0) * 1000, 1))

        max_acoustic = max(max_acoustic, a_res["acoustic_prob"])
        max_coercion = max(max_coercion, c_res["coercion_prob"])
        if verdict["fused_risk"] >= peak_risk:
            peak_risk = verdict["fused_risk"]
            final_level = verdict["alert_level"]

    return {
        "windows_scored": len(latencies_ms),
        "max_acoustic_prob": round(max_acoustic, 3),
        "max_coercion_prob": round(max_coercion, 3),
        "peak_fused_risk": round(peak_risk, 3),
        "verdict": final_level,
        "latencies_ms": latencies_ms,
    }


def main() -> None:
    models = FirewallModels()
    _ = models.score_acoustic(np.zeros(WIN_SAMPLES, dtype=np.float32))

    clips = sorted(
        p for p in DEMO_DIR.glob("*.wav")
        if not p.name.endswith("_raw.wav") and not p.name.endswith("_ref.wav")
    )
    if not clips:
        print("No .wav files found in demo_assets/.")
        return

    rows = []
    all_latencies = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for clip in clips:
            is_attack = any(k in clip.name.lower() for k in ("scam", "clone", "attack", "fake"))
            expected = "ATTACK" if is_attack else "SAFE"

            res = evaluate_clip(models, load_16k(clip))
            all_latencies.extend(res["latencies_ms"])
            detected = res["verdict"] in ("CAUTION", "CRITICAL_SCAM")
            passed = (detected == is_attack)

            rows.append({
                "clip": clip.name,
                "condition": "8kHz_mulaw" if "8khz" in clip.name.lower() else "clean_16kHz",
                "expected": expected,
                "verdict": res["verdict"],
                "peak_fused_risk": res["peak_fused_risk"],
                "max_acoustic": res["max_acoustic_prob"],
                "max_coercion": res["max_coercion_prob"],
                "avg_latency_ms": round(float(np.mean(res["latencies_ms"])), 1) if res["latencies_ms"] else 0.0,
                "passed": passed,
            })

            if "8khz" not in clip.name.lower() and not (DEMO_DIR / f"{clip.stem}_8khz.wav").exists():
                deg_path = Path(tmpdir) / f"{clip.stem}_8khz.wav"
                degrade_to_8khz_mulaw(clip, deg_path)
                deg_res = evaluate_clip(models, load_16k(deg_path))
                all_latencies.extend(deg_res["latencies_ms"])
                deg_detected = deg_res["verdict"] in ("CAUTION", "CRITICAL_SCAM")
                rows.append({
                    "clip": f"{clip.stem}_8khz (auto)",
                    "condition": "8kHz_mulaw",
                    "expected": expected,
                    "verdict": deg_res["verdict"],
                    "peak_fused_risk": deg_res["peak_fused_risk"],
                    "max_acoustic": deg_res["max_acoustic_prob"],
                    "max_coercion": deg_res["max_coercion_prob"],
                    "avg_latency_ms": round(float(np.mean(deg_res["latencies_ms"])), 1) if deg_res["latencies_ms"] else 0.0,
                    "passed": (deg_detected == is_attack),
                })

    summary = {
        "hardware": "CUDA GPU" if models.has_cuda else "Local CPU",
        "clips_evaluated": len(rows),
        "accuracy_pct": round(100.0 * sum(r["passed"] for r in rows) / len(rows), 1),
        "mean_window_latency_ms": round(float(np.mean(all_latencies)), 1) if all_latencies else 0.0,
        "p95_window_latency_ms": round(float(np.percentile(all_latencies, 95)), 1) if all_latencies else 0.0,
        "results": rows,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Hardware: {summary['hardware']} | Accuracy: {summary['accuracy_pct']}% | Mean Latency: {summary['mean_window_latency_ms']} ms | p95 Latency: {summary['p95_window_latency_ms']} ms\n")
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['clip']:<28} | {r['condition']:<11} | Risk={r['peak_fused_risk']:.3f} ({r['verdict']}) | {r['avg_latency_ms']} ms")
    print(f"\nSaved JSON report to: {OUT_FILE}")


if __name__ == "__main__":
    main()
