# backend/models/fusion.py
import numpy as np


class AudioRingBuffer:
    """Zero-retention in-memory sliding window buffer for 16 kHz PCM16 streams."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)
        self.hop_samples = int(sample_rate * hop_sec)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_eval = 0

    def append_pcm16(self, raw_bytes: bytes) -> np.ndarray | None:
        """Append raw 16-bit signed PCM bytes; return a 4s float32 window every 2s."""
        if len(raw_bytes) < 2:
            return None

        # Ensure an even byte count for 16-bit samples
        usable_len = len(raw_bytes) - (len(raw_bytes) % 2)
        chunk = (
            np.frombuffer(raw_bytes[:usable_len], dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        self.buffer = np.concatenate([self.buffer, chunk])[-self.window_samples :]
        self.samples_since_eval += len(chunk)

        if (
            len(self.buffer) >= self.window_samples
            and self.samples_since_eval >= self.hop_samples
        ):
            self.samples_since_eval = 0
            return self.buffer.copy()

        return None

    def purge(self) -> None:
        """Overwrite and clear audio samples from RAM when a call ends."""
        if self.buffer.size > 0:
            self.buffer.fill(0.0)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_eval = 0


class RiskFusionEngine:
    """Fuses acoustic deepfake and behavioral coercion scores with EMA smoothing."""

    def __init__(self, alpha: float = 0.55) -> None:
        self.alpha = alpha
        self.smoothed_risk = 0.0

    def update(self, acoustic_prob: float, coercion_prob: float) -> dict:
        a_prob = float(np.clip(acoustic_prob, 0.0, 1.0))
        c_prob = float(np.clip(coercion_prob, 0.0, 1.0))

        # Either high deepfake OR high coercion escalates risk;
        # both firing together triggers a synergy boost for rapid intervention.
        base_risk = 0.50 * a_prob + 0.50 * c_prob
        synergy = a_prob >= 0.45 and c_prob >= 0.55
        synergy_boost = 0.25 if synergy else 0.0

        instant_risk = min(
            1.0,
            max(base_risk + synergy_boost, a_prob * 0.85, c_prob * 0.85),
        )

        # Exponential Moving Average prevents single-chunk false alarms
        self.smoothed_risk = (
            self.alpha * instant_risk + (1.0 - self.alpha) * self.smoothed_risk
        )

        if self.smoothed_risk >= 0.72:
            level = "CRITICAL_SCAM"
        elif self.smoothed_risk >= 0.45:
            level = "CAUTION"
        else:
            level = "SAFE"

        return {
            "instant_risk": round(instant_risk, 3),
            "fused_risk": round(self.smoothed_risk, 3),
            "alert_level": level,
            "synergy_triggered": synergy,
        }

    def reset(self) -> None:
        self.smoothed_risk = 0.0
