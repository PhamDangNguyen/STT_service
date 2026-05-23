import asyncio
import io

import numpy as np
import soundfile as sf

from .base import PipelineStep
from .models import (
    VerificationInput,
    VerificationOutput,
    VerifiedSegment,
    DiarizedSegment,
)

SAMPLE_RATE = 16_000


class VerificationStep(PipelineStep[VerificationInput, VerificationOutput]):
    def __init__(self, embedding_model=None, device: str = "cpu"):
        self._model = embedding_model
        self._device = device

    async def run(self, input: VerificationInput) -> VerificationOutput:
        if not input.speaker_profiles or self._model is None:
            segments = self._assign_without_profiles(input.audio, input.segments)
            return VerificationOutput(segments=segments)

        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, self._verify, input.audio, input.segments, input.speaker_profiles
        )
        return VerificationOutput(segments=segments)

    def _extract_segment_audio(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        s = int(start * SAMPLE_RATE)
        e = int(end * SAMPLE_RATE)
        return audio[s:e]

    def _assign_without_profiles(
        self, audio: np.ndarray, segments: list[DiarizedSegment]
    ) -> list[VerifiedSegment]:
        return [
            VerifiedSegment(
                start=seg.start,
                end=seg.end,
                speaker_name=seg.speaker_label,
                audio=self._extract_segment_audio(audio, seg.start, seg.end),
            )
            for seg in segments
        ]

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        waveform = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return emb.squeeze().cpu().numpy()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _verify(
        self,
        audio: np.ndarray,
        segments: list[DiarizedSegment],
        speaker_profiles: dict[str, list[float]],
    ) -> list[VerifiedSegment]:
        profile_embeddings = {
            name: np.array(vec, dtype=np.float32)
            for name, vec in speaker_profiles.items()
        }
        label_to_name: dict[str, str] = {}
        result: list[VerifiedSegment] = []

        for seg in segments:
            seg_audio = self._extract_segment_audio(audio, seg.start, seg.end)
            if seg.speaker_label not in label_to_name:
                if len(seg_audio) > 0:
                    emb = self._embed(seg_audio)
                    best_name = max(
                        profile_embeddings,
                        key=lambda n: self._cosine_similarity(emb, profile_embeddings[n]),
                    )
                else:
                    best_name = seg.speaker_label
                label_to_name[seg.speaker_label] = best_name
            result.append(VerifiedSegment(
                start=seg.start,
                end=seg.end,
                speaker_name=label_to_name[seg.speaker_label],
                audio=seg_audio,
            ))
        return result
