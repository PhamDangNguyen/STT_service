from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: float
    duration: float
    text: str
    speaker: str
