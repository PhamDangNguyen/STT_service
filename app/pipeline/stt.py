import asyncio
import numpy as np

from .base import PipelineStep
from .models import STTInput, STTOutput, TranscribedSegment, VerifiedSegment


class STTStep(PipelineStep[STTInput, STTOutput]):
    def __init__(self, model, language: str = "vi"):
        self._model = model
        self._language = language

    async def run(self, input: STTInput) -> STTOutput:
        segments = input.segments
        total = max(len(segments), 1)
        loop = asyncio.get_event_loop()
        results: list[TranscribedSegment] = []

        for i, seg in enumerate(segments):
            text = await loop.run_in_executor(None, self._transcribe, seg.audio)
            duration = seg.end - seg.start
            results.append(TranscribedSegment(
                start=seg.start,
                duration=duration,
                text=text.strip(),
                speaker=seg.speaker_name,
            ))
            await input.progress_callback(i, total)

        return STTOutput(segments=results)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
        )
        return " ".join(seg.text for seg in segments)
