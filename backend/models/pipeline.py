# backend/models/pipeline.py
import os
import re
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel
from transformers import pipeline

COERCION_LABELS = [
    "urgent demand to transfer money or pay immediately",
    "impersonating bank, police, or government authority",
    "requesting one-time password, PIN, or remote screen access",
    "demanding secrecy or isolating the victim from family",
]

HIGH_RISK_PATTERNS = re.compile(
    r"\b(otp|one[- ]time password|cvv|anydesk|teamviewer|rustdesk|"
    r"wire transfer|gift card|upi pin|screen share|arrest warrant)\b",
    re.IGNORECASE,
)


class FirewallModels:
    """Loads all four call-firewall models once and scores 16 kHz float32 audio in memory."""

    def __init__(self) -> None:
        self.has_cuda = torch.cuda.is_available()
        hf_device = 0 if self.has_cuda else -1

        # Split CPU cores across the two concurrent branches (Wav2Vec2 vs. Whisper+NLI)
        cpu_threads = max(2, (os.cpu_count() or 4) // 2)
        if not self.has_cuda:
            torch.set_num_threads(cpu_threads)

        print(
            f"[Init] Device: {'CUDA GPU' if self.has_cuda else f'CPU ({cpu_threads} threads/branch)'}"
        )

        # 1. Silero VAD (ONNX runtime manages its own single-thread session)
        print("[Init] 1/4 Loading Silero VAD (ONNX)...")
        self.vad_model = load_silero_vad(onnx=True)

        # 2. Acoustic Deepfake Classifier (Wav2Vec2)
        print("[Init] 2/4 Loading Wav2Vec2 deepfake voice detector...")
        self.deepfake_pipe = pipeline(
            "audio-classification",
            model="garystafford/wav2vec2-deepfake-voice-detector",
            device=hf_device,
        )

        # 3. Streaming ASR (Faster-Whisper)
        asr_name = "distil-large-v3" if self.has_cuda else "base.en"
        print(f"[Init] 3/4 Loading Faster-Whisper ({asr_name})...")
        self.asr_model = WhisperModel(
            asr_name,
            device="cuda" if self.has_cuda else "cpu",
            compute_type="float16" if self.has_cuda else "int8",
            cpu_threads=cpu_threads,
        )

        # 4. Social-Engineering Zero-Shot NLI
        nli_name = (
            "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
            if self.has_cuda
            else "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
        )
        print(f"[Init] 4/4 Loading Zero-Shot NLI ({nli_name})...")
        self.nli_pipe = pipeline(
            "zero-shot-classification",
            model=nli_name,
            device=hf_device,
        )
        print("[Init] All 4 models loaded and ready.")

    @staticmethod
    def _ensure_mono_f32(audio_16k: np.ndarray) -> np.ndarray:
        if audio_16k.ndim > 1:
            audio_16k = audio_16k.mean(axis=1)
        return np.ascontiguousarray(audio_16k, dtype=np.float32)

    def has_speech(self, audio_16k: np.ndarray) -> bool:
        clean = self._ensure_mono_f32(audio_16k)
        tensor = torch.from_numpy(clean)
        timestamps = get_speech_timestamps(tensor, self.vad_model, sampling_rate=16000)
        return len(timestamps) > 0

    def score_acoustic(self, audio_16k: np.ndarray) -> dict:
        clean = self._ensure_mono_f32(audio_16k)
        with torch.inference_mode():
            preds = self.deepfake_pipe({"raw": clean, "sampling_rate": 16000})

        fake_prob = 0.0
        for item in preds:
            if any(k in item["label"].lower() for k in ("fake", "spoof", "ai", "synthetic")):
                fake_prob = float(item["score"])
                break

        return {
            "acoustic_prob": round(fake_prob, 3),
            "raw_labels": preds,
        }

    def score_coercion(self, audio_16k: np.ndarray) -> dict:
        clean = self._ensure_mono_f32(audio_16k)
        segments, _ = self.asr_model.transcribe(clean, beam_size=1, vad_filter=False)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()

        if len(transcript.split()) < 3:
            return {
                "transcript": transcript,
                "coercion_prob": 0.0,
                "triggers": [],
                "label_scores": {},
            }

        with torch.inference_mode():
            nli_out = self.nli_pipe(transcript, COERCION_LABELS, multi_label=True)

        label_scores = {
            lbl: round(float(sc), 3)
            for lbl, sc in zip(nli_out["labels"], nli_out["scores"])
        }
        top_score = float(max(nli_out["scores"]))
        triggers = [lbl for lbl, sc in label_scores.items() if sc >= 0.55]

        # Deterministic guardrail for explicit OTP / remote-access / payment theft
        regex_hits = HIGH_RISK_PATTERNS.findall(transcript)
        if regex_hits:
            top_score = max(top_score, 0.90)
            unique_hits = sorted(set(hit.lower() for hit in regex_hits))
            triggers.append(f"keyword: {', '.join(unique_hits)}")

        return {
            "transcript": transcript,
            "coercion_prob": round(top_score, 3),
            "triggers": triggers,
            "label_scores": label_scores,
        }
