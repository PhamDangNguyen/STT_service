from ..exceptions import PipelineStepError
from ..logging_setup import get_logger
from ..repositories.note_repository import NoteRepository
from ..stores.base import ProgressStore
from ..pipeline.runner import PipelineRunner
from ..pipeline.models import PipelineInput

logger = get_logger(__name__)


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
            await self._progress_store.finish(note_id)
            logger.info("Transcription done | note_id=%s segments=%d", note_id, len(segments))
        except Exception as exc:
            logger.error("Transcription failed | note_id=%s error=%s", note_id, exc, exc_info=True)
            await self._progress_store.fail(note_id, str(exc))
            raise
