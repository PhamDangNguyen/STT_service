from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from .segment import TranscriptSegment


class NoteDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    audio_url: str
    transcript: list[TranscriptSegment] = []
    summary: str = ""
    mindmap: str = ""
    slide_url: str = ""
    suggestions: list[dict] = []
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
