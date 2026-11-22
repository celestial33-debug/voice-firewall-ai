# test_fusion.py
import numpy as np
from backend.models.fusion import AudioRingBuffer, RiskFusionEngine


def test_ring_buffer() -> None:
    print("=== 1. Testing AudioRingBuffer (4s window, 2s hop) ===")
    ring = AudioRingBuffer(sample_rate=16000, window_sec=4.0, hop_sec=2.0)

    # Stream 8.0 seconds of audio in 0.5-second PCM16 chunks (16 packets)
    chunk_samples = 8000  # 0.5s at 16 kHz
    dummy_pcm = (np.zeros(chunk_samples, dtype=np.int16)).tobytes()

    emitted_at_sec = []
    for i in range(1, 17):
        window = ring.append_pcm16(dummy_pcm)
        elapsed = i * 0.5
        if window is not None:
            emitted_at_sec.append(elapsed)
            assert window.shape == (64000,), f"Expected 64000 samples, got {window.shape}"
            assert window.dtype == np.float32, f"Expected float32, got {window.dtype}"
            print(f"  t={elapsed:4.1f}s -> Emitted window | shape={window.shape} | dtype={window.dtype}")

    assert emitted_at_sec == [4.0, 6.0, 8.0], f"Unexpected emit times: {emitted_at_sec}"

    ring.purge()
    assert len(ring.buffer) == 0, "Buffer was not cleared after purge()"
    print("  Purge check passed: 0 samples retained in RAM.\n")


def test_risk_fusion() -> None:
    print("=== 2. Testing RiskFusionEngine (EMA + Synergy) ===")
    fusion = RiskFusionEngine(alpha=0.55)

    simulated_call = [
        ("Normal greeting",           0.08, 0.05, "SAFE"),
        ("Excited normal banter",     0.12, 0.15, "SAFE"),
        ("Brief background glitch",   0.50, 0.10, "SAFE"),
        ("Cloned bank pretext",       0.78, 0.62, "CAUTION"),
        ("Cloned voice demands OTP",  0.88, 0.92, "CRITICAL_SCAM"),
        ("Sustained coercion attack", 0.91, 0.95, "CRITICAL_SCAM"),
    ]

    for idx, (label, a_prob, c_prob, expected_level) in enumerate(simulated_call, start=1):
        res = fusion.update(a_prob, c_prob)
        print(
            f"  Win {idx} ({label:<25}) | "
            f"Acoustic={a_prob:.2f} Coercion={c_prob:.2f} -> "
            f"Fused={res['fused_risk']:.3f} [{res['alert_level']}] "
            f"(Synergy={res['synergy_triggered']})"
        )
        assert res["alert_level"] == expected_level, (
            f"Window {idx}: expected {expected_level}, got {res['alert_level']}"
        )

    print("\nALL STEP 3 TESTS PASSED!")


if __name__ == "__main__":
    test_ring_buffer()
    test_risk_fusion()
