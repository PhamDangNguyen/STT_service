import asyncio
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np

from ..exceptions import AudioFetchError, UnsupportedAudioFormatError
from .base import PipelineStep
from .models import AudioLoadInput, AudioLoadOutput, AudioArray

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus"})
SAMPLE_RATE = 16_000


class AudioLoaderStep(PipelineStep[AudioLoadInput, AudioLoadOutput]):
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client or httpx.AsyncClient(timeout=120.0)

    async def run(self, input: AudioLoadInput) -> AudioLoadOutput:
        ext = Path(input.audio_url.split("?")[0]).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedAudioFormatError(ext)

        try:
            response = await self._client.get(input.audio_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AudioFetchError(input.audio_url, str(e)) from e

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as src_f:
            src_f.write(response.content)
            src_path = src_f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst_f:
            dst_path = dst_f.name

        try:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None, self._convert_and_load, src_path, dst_path
            )
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(dst_path).unlink(missing_ok=True)

        duration = len(audio) / SAMPLE_RATE
        return AudioLoadOutput(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            duration_seconds=duration,
            source_extension=ext,
        )

    def _convert_and_load(self, src_path: str, dst_path: str) -> AudioArray:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le",
            dst_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise AudioFetchError(src_path, result.stderr.decode())
        raw = Path(dst_path).read_bytes()
        return np.frombuffer(raw, dtype=np.float32).copy()
