import asyncio
import numpy as np

from .base import PipelineStep
from .models import VADInput, VADOutput, AudioChunk

SAMPLE_RATE = 16_000


class VADStep(PipelineStep[VADInput, VADOutput]):
    def __init__(self, model, utils, device: str = "cpu", chunk_max_seconds: float = 300.0):
        self._model = model
        self._device = device
        self._chunk_max_seconds = chunk_max_seconds
        if utils is not None:
            self._get_speech_ts, *_ = utils
        else:
            self._get_speech_ts = None

    async def run(self, input: VADInput) -> VADOutput:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._run_vad, input.audio)
        return VADOutput(chunks=chunks)

    def _run_vad(self, audio: np.ndarray) -> list[AudioChunk]:
        if self._model is None or self._get_speech_ts is None:
            return [AudioChunk(start_sec=0.0, end_sec=len(audio) / SAMPLE_RATE, audio=audio)]

        import torch
        tensor = torch.from_numpy(audio).to(self._device)
        speech_timestamps = self._get_speech_ts(tensor, self._model, sampling_rate=SAMPLE_RATE)
        if not speech_timestamps:
            return []
        return self._group_into_chunks(speech_timestamps, audio)

    def _group_into_chunks(self, speech_timestamps: list[dict], audio: np.ndarray) -> list[AudioChunk]:
        max_samples = int(self._chunk_max_seconds * SAMPLE_RATE)
        chunks: list[AudioChunk] = []
        current_window: list[dict] = []
        current_speech_samples = 0

        for ts in speech_timestamps:
            seg_samples = ts["end"] - ts["start"]
            if current_window and current_speech_samples + seg_samples > max_samples:
                chunks.append(self._make_chunk(current_window, audio))
                current_window = []
                current_speech_samples = 0
            current_window.append(ts)
            current_speech_samples += seg_samples

        if current_window:
            chunks.append(self._make_chunk(current_window, audio))

        return chunks

    @staticmethod
    def _make_chunk(window: list[dict], audio: np.ndarray) -> AudioChunk:
        start = window[0]["start"]
        end = window[-1]["end"]
        return AudioChunk(
            start_sec=start / SAMPLE_RATE,
            end_sec=end / SAMPLE_RATE,
            audio=audio[start:end],
        )
