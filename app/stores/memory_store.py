from datetime import datetime, timedelta, timezone
from ..models.progress import ProgressRecord, ProgressStatus


class InMemoryProgressStore:
    """
    Class help update and manage progress in service. Auto remove progress record after ttl_seconds since it is marked as done or error.
     - init(note_id): create new progress record with status pending and progress 0
     - update(note_id, progress, step): update progress record with status processing, given progress and step
     - finish(note_id): update progress record with status done and progress 100
     - fail(note_id, error): update progress record with status error and given error message
     - get(note_id): get progress record by note_id, return None if not found or expired
     - _evict_expired(): internal method to remove expired progress records
    """
    def __init__(self, ttl_seconds: int = 3600):
        self._records: dict[str, ProgressRecord] = {}
        self._finalized_at: dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    async def init(self, note_id: str) -> None:
        self._records[note_id] = ProgressRecord(
            note_id=note_id,
            status=ProgressStatus.PENDING,
            progress=0,
            updated_at=datetime.now(timezone.utc),
        )
        self._finalized_at.pop(note_id, None)

    async def update(self, note_id: str, progress: int, step: str) -> None:
        rec = self._records.get(note_id)
        if rec is None:
            return
        self._records[note_id] = rec.model_copy(update={
            "status": ProgressStatus.PROCESSING,
            "progress": progress,
            "current_step": step,
            "updated_at": datetime.now(timezone.utc),
        })

    async def finish(self, note_id: str) -> None:
        rec = self._records.get(note_id)
        if rec is None:
            return
        now = datetime.now(timezone.utc)
        self._records[note_id] = rec.model_copy(update={
            "status": ProgressStatus.DONE,
            "progress": 100,
            "current_step": None,
            "updated_at": now,
        })
        self._finalized_at[note_id] = now

    async def fail(self, note_id: str, error: str) -> None:
        rec = self._records.get(note_id)
        if rec is None:
            return
        now = datetime.now(timezone.utc)
        self._records[note_id] = rec.model_copy(update={
            "status": ProgressStatus.ERROR,
            "error": error,
            "updated_at": now,
        })
        self._finalized_at[note_id] = now

    async def get(self, note_id: str) -> ProgressRecord | None:
        self._evict_expired()
        return self._records.get(note_id)

    def _evict_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            nid for nid, ts in self._finalized_at.items()
            if now - ts > self._ttl
        ]
        for nid in expired:
            self._records.pop(nid, None)
            self._finalized_at.pop(nid, None)
