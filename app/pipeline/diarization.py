import asyncio
import io

import numpy as np
import soundfile as sf

from .base import PipelineStep
from .models import DiarizationInput, DiarizationOutput, DiarizedSegment

SAMPLE_RATE = 16_000


class DiarizationStep(PipelineStep[DiarizationInput, DiarizationOutput]):
    def __init__(self, pipeline):
        self._pipeline = pipeline

    async def run(self, input: DiarizationInput) -> DiarizationOutput:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, self._run_diarization, input.audio, input.num_speakers
        )
        return DiarizationOutput(segments=segments)

    def _run_diarization(
        self, audio: np.ndarray, num_speakers: int | None
    ) -> list[DiarizedSegment]:
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        diarization = self._pipeline(buf, **kwargs)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                speaker_label=speaker,
            ))
        return segments
