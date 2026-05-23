import time

from ..logging_setup import get_logger
from ..stores.base import ProgressStore
from ..models.segment import TranscriptSegment

logger = get_logger(__name__)
from .models import (
    PipelineInput,
    AudioLoadInput,
    VADInput,
    DiarizationInput,
    VerificationInput,
    STTInput,
    LLMNormInput,
)
from .audio_loader import AudioLoaderStep
from .vad import VADStep
from .diarization import DiarizationStep
from .verification import VerificationStep
from .stt import STTStep
from .llm_norm import LLMNormStep


class PipelineRunner:
    def __init__(
        self,
        audio_loader: AudioLoaderStep,
        vad: VADStep,
        diarization: DiarizationStep,
        verification: VerificationStep,
        stt: STTStep,
        llm_norm: LLMNormStep,
        progress_store: ProgressStore,
    ):
        self._audio_loader = audio_loader
        self._vad = vad
        self._diarization = diarization
        self._verification = verification
        self._stt = stt
        self._llm_norm = llm_norm
        self._progress_store = progress_store

    async def run(self, pipeline_input: PipelineInput) -> list[TranscriptSegment]:
        note_id = pipeline_input.note_id
        ps = self._progress_store
        pipeline_start = time.perf_counter()
        logger.info("Pipeline start | note_id=%s", note_id)

        # Step 1: Load audio (10%)
        await ps.update(note_id, 5, "AudioLoaderStep")
        t0 = time.perf_counter()
        logger.debug("Step AudioLoaderStep begin | note_id=%s", note_id)
        audio_out = await self._audio_loader.run(
            AudioLoadInput(note_id=note_id, audio_url=pipeline_input.audio_url)
        )
        logger.info("Step AudioLoaderStep done | note_id=%s elapsed=%.2fs", note_id, time.perf_counter() - t0)
        await ps.update(note_id, 10, "AudioLoaderStep")

        # Step 2: VAD (20%)
        await ps.update(note_id, 15, "VADStep")
        t0 = time.perf_counter()
        logger.debug("Step VADStep begin | note_id=%s", note_id)
        vad_out = await self._vad.run(
            VADInput(audio=audio_out.audio, sample_rate=audio_out.sample_rate)
        )
        logger.info("Step VADStep done | note_id=%s elapsed=%.2fs", note_id, time.perf_counter() - t0)
        await ps.update(note_id, 20, "VADStep")

        # Step 3: Diarization (40%)
        await ps.update(note_id, 30, "DiarizationStep")
        t0 = time.perf_counter()
        logger.debug("Step DiarizationStep begin | note_id=%s", note_id)
        diar_out = await self._diarization.run(
            DiarizationInput(audio=audio_out.audio, sample_rate=audio_out.sample_rate)
        )
        logger.info(
            "Step DiarizationStep done | note_id=%s segments=%d elapsed=%.2fs",
            note_id, len(diar_out.segments), time.perf_counter() - t0,
        )
        await ps.update(note_id, 40, "DiarizationStep")

        # Step 4: Verification (50%)
        await ps.update(note_id, 45, "VerificationStep")
        t0 = time.perf_counter()
        logger.debug("Step VerificationStep begin | note_id=%s", note_id)
        ver_out = await self._verification.run(
            VerificationInput(
                audio=audio_out.audio,
                sample_rate=audio_out.sample_rate,
                segments=diar_out.segments,
                speaker_profiles=pipeline_input.speaker_profiles,
            )
        )
        logger.info("Step VerificationStep done | note_id=%s elapsed=%.2fs", note_id, time.perf_counter() - t0)
        await ps.update(note_id, 50, "VerificationStep")

        # Step 5: STT (50-85%) — per-chunk progress callback
        t0 = time.perf_counter()
        logger.debug("Step STTStep begin | note_id=%s", note_id)

        async def stt_progress(chunk_index: int, total_chunks: int) -> None:
            processed = chunk_index + 1
            progress = 50 + round(processed / max(total_chunks, 1) * 35)
            progress = min(progress, 85)
            logger.debug(
                "STTStep progress | note_id=%s chunk=%d/%d progress=%d%%",
                note_id, processed, total_chunks, progress,
            )
            await ps.update(note_id, progress, "STTStep")

        stt_out = await self._stt.run(
            STTInput(
                segments=ver_out.segments,
                sample_rate=audio_out.sample_rate,
                progress_callback=stt_progress,
            )
        )
        logger.info(
            "Step STTStep done | note_id=%s segments=%d elapsed=%.2fs",
            note_id, len(stt_out.segments), time.perf_counter() - t0,
        )

        # Step 6: LLM normalization (100%)
        await ps.update(note_id, 90, "LLMNormStep")
        t0 = time.perf_counter()
        logger.debug("Step LLMNormStep begin | note_id=%s", note_id)
        norm_out = await self._llm_norm.run(LLMNormInput(segments=stt_out.segments))
        logger.info("Step LLMNormStep done | note_id=%s elapsed=%.2fs", note_id, time.perf_counter() - t0)

        total_elapsed = time.perf_counter() - pipeline_start
        result = [
            TranscriptSegment(
                start=s.start,
                duration=s.duration,
                text=s.text,
                speaker=s.speaker,
            )
            for s in norm_out.segments
        ]
        logger.info(
            "Pipeline complete | note_id=%s total_segments=%d total_elapsed=%.2fs",
            note_id, len(result), total_elapsed,
        )
        return result
