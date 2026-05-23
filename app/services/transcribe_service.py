import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..exceptions import PipelineStepError
from ..logging_setup import get_logger
from ..repositories.note_repository import NoteRepository
from ..stores.base import ProgressStore
from ..pipeline.runner import PipelineRunner
from ..pipeline.models import PipelineInput

logger = get_logger(__name__)

_OUTPUTS_DIR = Path("outputs")


async def _save_output(note_id: str, segments: list, audio_url: str) -> None:
    now = datetime.now(timezone.utc)
    # Reverse epoch so newest file sorts first alphabetically
    reverse_ts = 9_999_999_999 - int(time.time())
    base_name = f"{reverse_ts:010d}_{note_id}"

    folder = _OUTPUTS_DIR / base_name
    folder.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = folder / f"{base_name}.json"
    payload = {
        "note_id": note_id,
        "created_at": now.isoformat(),
        "segments": [s.model_dump() for s in segments],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON saved | path=%s", json_path)

    # Save audio alongside JSON with the same base name
    ext = Path(audio_url.split("?")[0]).suffix.lower() or ".audio"
    audio_path = folder / f"{base_name}{ext}"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()
        audio_path.write_bytes(response.content)
        logger.info("Audio saved | path=%s", audio_path)
    except Exception as e:
        logger.warning("Audio save failed | note_id=%s error=%s", note_id, e)


class TranscribeService:
    def __init__(
        self,
        note_repo: NoteRepository,
        pipeline: PipelineRunner,
        progress_store: ProgressStore,
    ):
        self._repo = note_repo
        self._pipeline = pipeline
        self._progress_store = progress_store

    async def transcribe(self, note_id: str) -> None:
        logger.info("Transcription started | note_id=%s", note_id)
        try:
            note = await self._repo.get_by_id(note_id)
            logger.debug("Note fetched | note_id=%s audio_url=%s", note_id, note.audio_url)
            pipeline_input = PipelineInput(
                note_id=note_id,
                audio_url=note.audio_url,
            )
            segments = await self._pipeline.run(pipeline_input)
            logger.debug("Pipeline finished | note_id=%s segments=%d", note_id, len(segments))
            await self._repo.update_transcript(note_id, segments)
            await _save_output(note_id, segments, note.audio_url)
            await self._progress_store.finish(note_id)
            logger.info("Transcription done | note_id=%s segments=%d", note_id, len(segments))
        except Exception as exc:
            logger.error("Transcription failed | note_id=%s error=%s", note_id, exc, exc_info=True)
            await self._progress_store.fail(note_id, str(exc))
            raise
