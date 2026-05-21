from ..exceptions import PipelineStepError
from ..repositories.note_repository import NoteRepository
from ..stores.base import ProgressStore
from ..pipeline.runner import PipelineRunner
from ..pipeline.models import PipelineInput


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
        try:
            note = await self._repo.get_by_id(note_id)
            pipeline_input = PipelineInput(
                note_id=note_id,
                audio_url=note.audio_url,
            )
            segments = await self._pipeline.run(pipeline_input)
            await self._repo.update_transcript(note_id, segments)
            await self._progress_store.finish(note_id)
        except Exception as exc:
            await self._progress_store.fail(note_id, str(exc))
            raise
