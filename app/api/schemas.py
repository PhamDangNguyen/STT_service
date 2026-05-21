from pydantic import BaseModel
from ..models.progress import ProgressRecord


class TranscribeRequest(BaseModel):
    note_id: str


class TranscribeResponse(BaseModel):
    note_id: str
    status_url: str


class StatusResponse(BaseModel):
    note_id: str
    status: str
    progress: int
    current_step: str | None = None
    error: str | None = None
