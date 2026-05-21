from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class ProgressStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class ProgressRecord(BaseModel):
    note_id: str
    status: ProgressStatus
    progress: int  # 0-100
    current_step: str | None = None
    error: str | None = None
    updated_at: datetime
