from dataclasses import dataclass, field
from typing import Any
from motor.motor_asyncio import AsyncIOMotorClient
from .config import Settings
from .stores.memory_store import InMemoryProgressStore


@dataclass
class AppState:
    settings: Settings
    db_client: AsyncIOMotorClient
    progress_store: InMemoryProgressStore
    vad_model: Any = None
    vad_utils: Any = None
    diarization_pipeline: Any = None
    embedding_model: Any = None
    whisper_model: Any = None
