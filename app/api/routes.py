from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from ..exceptions import ProgressNotFoundError
from ..stores.base import ProgressStore
from ..services.transcribe_service import TranscribeService
from .deps import get_progress_store, get_transcribe_service
from .schemas import TranscribeRequest, TranscribeResponse, StatusResponse

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


@router.post("", status_code=202, response_model=TranscribeResponse)
async def start_transcription(
    body: TranscribeRequest,
    background_tasks: BackgroundTasks,
    progress_store: Annotated[ProgressStore, Depends(get_progress_store)],
    service: Annotated[TranscribeService, Depends(get_transcribe_service)],
) -> TranscribeResponse:
    await progress_store.init(body.note_id)
    background_tasks.add_task(service.transcribe, body.note_id)
    return TranscribeResponse(
        note_id=body.note_id,
        status_url=f"/api/transcribe/{body.note_id}/status",
    )


@router.get("/{note_id}/status", response_model=StatusResponse)
async def get_status(
    note_id: str,
    response: Response,
    progress_store: Annotated[ProgressStore, Depends(get_progress_store)],
) -> StatusResponse:
    response.headers["Cache-Control"] = "no-store"
    record = await progress_store.get(note_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription job found for note_id={note_id}",
        )
    return StatusResponse(
        note_id=record.note_id,
        status=record.status,
        progress=record.progress,
        current_step=record.current_step,
        error=record.error,
    )
