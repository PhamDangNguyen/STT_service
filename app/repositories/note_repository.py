from datetime import datetime, timezone
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..exceptions import NoteNotFoundError
from ..models.note import NoteDocument
from ..models.segment import TranscriptSegment


class NoteRepository:
    def __init__(self, db: AsyncIOMotorDatabase, collection: str = "notes"):
        self._col = db[collection]

    async def get_by_id(self, note_id: str) -> NoteDocument:
        doc = await self._col.find_one({"_id": note_id})
        if doc is None:
            # also try as ObjectId
            try:
                doc = await self._col.find_one({"_id": ObjectId(note_id)})
            except Exception:
                pass
        if doc is None:
            raise NoteNotFoundError(note_id)
        doc["_id"] = str(doc["_id"])
        return NoteDocument.model_validate(doc)

    async def update_transcript(
        self, note_id: str, segments: list[TranscriptSegment]
    ) -> None:
        now = datetime.now(timezone.utc)
        result = await self._col.update_one(
            {"_id": note_id},
            {"$set": {
                "transcript": [s.model_dump() for s in segments],
                "updatedAt": now,
            }},
        )
        if result.matched_count == 0:
            # try ObjectId
            try:
                result = await self._col.update_one(
                    {"_id": ObjectId(note_id)},
                    {"$set": {
                        "transcript": [s.model_dump() for s in segments],
                        "updatedAt": now,
                    }},
                )
            except Exception:
                pass
        if result.matched_count == 0:
            raise NoteNotFoundError(note_id)
