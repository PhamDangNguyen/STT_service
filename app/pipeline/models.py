from typing import TypeAlias, Callable, Awaitable
import numpy as np
from pydantic import BaseModel, ConfigDict

AudioArray: TypeAlias = np.ndarray


class _NpModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


# AudioLoaderStep
class AudioLoadInput(BaseModel):
    note_id: str
    audio_url: str


class AudioLoadOutput(_NpModel):
    audio: AudioArray        # float32, 16 kHz, mono
    sample_rate: int         # always 16000
    duration_seconds: float
    source_extension: str    # e.g. ".wav"


# VADStep
class VADInput(_NpModel):
    audio: AudioArray
    sample_rate: int


class AudioChunk(_NpModel):
    start_sec: float
    end_sec: float
    audio: AudioArray


class VADOutput(_NpModel):
    chunks: list[AudioChunk]


# DiarizationStep
class DiarizationInput(_NpModel):
    chunks: list[AudioChunk]
    sample_rate: int
    num_speakers: int | None = None


class DiarizedSegment(BaseModel):
    start: float
    end: float
    speaker_label: str   # e.g. "SPEAKER_00"


class DiarizationOutput(BaseModel):
    segments: list[DiarizedSegment]


# VerificationStep
class VerificationInput(_NpModel):
    audio: AudioArray
    sample_rate: int
    segments: list[DiarizedSegment]
    # speaker_name -> embedding vector; empty = skip verification
    speaker_profiles: dict[str, list[float]] = {}


class VerifiedSegment(_NpModel):
    start: float
    end: float
    speaker_name: str
    audio: AudioArray


class VerificationOutput(_NpModel):
    segments: list[VerifiedSegment]


# STTStep
class STTInput(_NpModel):
    segments: list[VerifiedSegment]
    sample_rate: int
    progress_callback: Callable[[int, int], Awaitable[None]]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TranscribedSegment(BaseModel):
    start: float
    duration: float
    text: str
    speaker: str


class STTOutput(BaseModel):
    segments: list[TranscribedSegment]


# LLMNormStep
class LLMNormInput(BaseModel):
    segments: list[TranscribedSegment]


class LLMNormOutput(BaseModel):
    segments: list[TranscribedSegment]


# Top-level PipelineInput
class PipelineInput(BaseModel):
    note_id: str
    audio_url: str
    speaker_profiles: dict[str, list[float]] = {}
