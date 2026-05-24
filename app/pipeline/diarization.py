import asyncio
import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from ..logging_setup import get_logger
from .base import PipelineStep
from .models import DiarizationInput, DiarizationOutput, DiarizedSegment, AudioChunk

logger = get_logger(__name__)

SAMPLE_RATE = 16_000
_MIN_EMBED_SAMPLES = SAMPLE_RATE // 2  # 0.5 s minimum for a reliable embedding


@dataclass
class _SpeakerEntry:
    """One entry in the cross-chunk global speaker registry."""
    name: str
    embedding: np.ndarray
    count: int = 1  # number of chunks that contributed to this embedding


class DiarizationStep(PipelineStep[DiarizationInput, DiarizationOutput]):
    """
    Runs pyannote diarization on each audio chunk independently, then stitches
    the per-chunk local speaker labels into a consistent global set using
    embedding cosine similarity.

    For each chunk:
      1. pyannote assigns local labels (SPEAKER_0, SPEAKER_1, …) with no
         cross-chunk identity — SPEAKER_0 in chunk N may be a different person
         than SPEAKER_0 in chunk N+1.
      2. For every local label we extract a representative speaker embedding.
      3. The embedding is compared against the global registry:
         • similarity >= threshold  → reuse the existing global speaker and
           update its embedding with a running mean.
         • similarity <  threshold  → register a new global speaker.
      4. All segments in the chunk are remapped to the resolved global labels.
    """

    def __init__(
        self,
        pipeline,
        embedding_model=None,
        device: str = "cpu",
        similarity_threshold: float = 0.75,
    ):
        self._pipeline = pipeline
        self._embedding_model = embedding_model
        self._device = device
        self._similarity_threshold = similarity_threshold

    async def run(self, input: DiarizationInput) -> DiarizationOutput:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, self._run_chunked, input.chunks, input.num_speakers
        )
        return DiarizationOutput(segments=segments)

    # ------------------------------------------------------------------ #
    # Per-chunk pyannote                                                   #
    # ------------------------------------------------------------------ #

    def _diarize_chunk(
        self, audio: np.ndarray, num_speakers: int | None
    ) -> list[DiarizedSegment]:
        """Run pyannote on a single audio chunk. Returns segments with LOCAL
        speaker labels (0-based, relative to this chunk only)."""
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
        diarization = self._pipeline(buf, **kwargs)
        annotation = (
            diarization.speaker_diarization
            if hasattr(diarization, "speaker_diarization")
            else diarization
        )
        return [
            DiarizedSegment(start=turn.start, end=turn.end, speaker_label=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        # Inference wrapper expects {"waveform": [1, T], "sample_rate": int} on CPU
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        result = self._embedding_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vec = np.array(result, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _representative_embedding(
        self,
        chunk_audio: np.ndarray,
        local_segs: list[DiarizedSegment],
        label: str,
    ) -> np.ndarray | None:
        """Concatenate all audio for `label` in this chunk and embed it.
        Returns None if there is not enough audio for a reliable embedding."""
        parts = [
            chunk_audio[int(seg.start * SAMPLE_RATE): int(seg.end * SAMPLE_RATE)]
            for seg in local_segs
            if seg.speaker_label == label
            and int(seg.end * SAMPLE_RATE) > int(seg.start * SAMPLE_RATE)
        ]
        if not parts:
            return None
        combined = np.concatenate(parts)
        if len(combined) < _MIN_EMBED_SAMPLES:
            return None
        return self._embed(combined)

    # ------------------------------------------------------------------ #
    # Global speaker registry                                              #
    # ------------------------------------------------------------------ #

    def _match_or_create(
        self,
        emb: np.ndarray,
        registry: list[_SpeakerEntry],
        counter: list[int],
    ) -> str:
        """Return the global speaker name whose embedding best matches `emb`.

        If the best cosine similarity is below `_similarity_threshold`, this
        is treated as a new speaker — a new entry is added to the registry.

        The matched entry's embedding is updated with a proper running mean so
        that every contributing chunk is weighted equally (not biased toward
        the most recent one).
        """
        best_entry: _SpeakerEntry | None = None
        best_sim = -1.0

        for entry in registry:
            sim = self._cosine_similarity(emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self._similarity_threshold:
            # Running mean: each of the N contributing embeddings has equal weight.
            n = best_entry.count
            avg = (best_entry.embedding * n + emb) / (n + 1)
            norm = np.linalg.norm(avg)
            best_entry.embedding = avg / norm if norm > 0 else avg
            best_entry.count += 1
            logger.info(
                "Speaker matched: → %s  (sim=%.3f >= threshold=%.2f, contributions=%d)",
                best_entry.name, best_sim, self._similarity_threshold, best_entry.count,
            )
            return best_entry.name

        # No match above threshold → new global speaker
        new_name = f"SPEAKER_{counter[0]:02d}"
        counter[0] += 1
        registry.append(_SpeakerEntry(name=new_name, embedding=emb))
        logger.info(
            "New speaker registered: %s  (best_sim=%.3f < threshold=%.2f)",
            new_name, best_sim, self._similarity_threshold,
        )
        return new_name

    # ------------------------------------------------------------------ #
    # Main entry                                                           #
    # ------------------------------------------------------------------ #

    def _run_chunked(
        self,
        chunks: list[AudioChunk],
        num_speakers: int | None,
    ) -> list[DiarizedSegment]:
        registry: list[_SpeakerEntry] = []
        counter = [0]  # mutable int so _match_or_create can increment it
        all_segments: list[DiarizedSegment] = []

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.audio) == 0:
                continue

            local_segs = self._diarize_chunk(chunk.audio, num_speakers)
            if not local_segs:
                logger.debug(
                    "Chunk %d (%.1fs–%.1fs): no segments returned by pyannote, skipping",
                    chunk_idx, chunk.start_sec, chunk.end_sec,
                )
                continue

            # Sort for determinism — set iteration order is undefined in Python
            local_labels = sorted({seg.speaker_label for seg in local_segs})
            logger.debug(
                "Chunk %d (%.1fs–%.1fs): pyannote returned %d local speaker(s): %s",
                chunk_idx, chunk.start_sec, chunk.end_sec,
                len(local_labels), local_labels,
            )

            local_to_global: dict[str, str] = {}

            for label in local_labels:
                if self._embedding_model is not None:
                    emb = self._representative_embedding(chunk.audio, local_segs, label)
                else:
                    emb = None

                if emb is not None:
                    global_name = self._match_or_create(emb, registry, counter)
                else:
                    # Embedding model absent or audio too short for this label.
                    # We cannot compare with the registry, so assign a new unique name.
                    global_name = f"SPEAKER_{counter[0]:02d}"
                    counter[0] += 1
                    logger.debug(
                        "Chunk %d, label %s: embedding unavailable → new speaker %s",
                        chunk_idx, label, global_name,
                    )

                local_to_global[label] = global_name

            logger.debug(
                "Chunk %d label mapping: %s",
                chunk_idx,
                ", ".join(f"{loc} → {glb}" for loc, glb in local_to_global.items()),
            )

            for seg in local_segs:
                all_segments.append(DiarizedSegment(
                    start=chunk.start_sec + seg.start,
                    end=chunk.start_sec + seg.end,
                    speaker_label=local_to_global[seg.speaker_label],
                ))

        unique_speakers = sorted({s.speaker_label for s in all_segments})
        logger.info(
            "Diarization complete: %d segments, %d unique speaker(s): %s",
            len(all_segments), len(unique_speakers), unique_speakers,
        )
        return all_segments
