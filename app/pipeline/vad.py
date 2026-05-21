import asyncio
import numpy as np

from .base import PipelineStep
from .models import VADInput, VADOutput, AudioChunk

SAMPLE_RATE = 16_000


class VADStep(PipelineStep[VADInput, VADOutput]):
    def __init__(self, model, utils):
        self._model = model
        self._get_speech_ts, *_ = utils

    async def run(self, input: VADInput) -> VADOutput:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._run_vad, input.audio)
        return VADOutput(chunks=chunks)

    def _run_vad(self, audio: np.ndarray) -> list[AudioChunk]:
        import torch
        tensor = torch.from_numpy(audio)
        speech_timestamps = self._get_speech_ts(
            tensor, self._model, sampling_rate=SAMPLE_RATE
        )
        result = []
        for ts in speech_timestamps:
            start = ts["start"]
            end = ts["end"]
            result.append(AudioChunk(
                start_sec=start / SAMPLE_RATE,
                end_sec=end / SAMPLE_RATE,
                audio=audio[start:end],
            ))
        return result
