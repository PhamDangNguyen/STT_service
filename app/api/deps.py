from typing import Annotated
from fastapi import Depends, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..config import Settings
from ..state import AppState
from ..stores.base import ProgressStore
from ..stores.memory_store import InMemoryProgressStore
from ..repositories.note_repository import NoteRepository
from ..pipeline.audio_loader import AudioLoaderStep
from ..pipeline.vad import VADStep
from ..pipeline.diarization import DiarizationStep
from ..pipeline.verification import VerificationStep
from ..pipeline.stt import STTStep
from ..pipeline.llm_norm import LLMNormStep
from ..pipeline.runner import PipelineRunner
from ..services.transcribe_service import TranscribeService


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_settings(state: Annotated[AppState, Depends(get_app_state)]) -> Settings:
    return state.settings


def get_progress_store(
    state: Annotated[AppState, Depends(get_app_state)],
) -> ProgressStore:
    return state.progress_store


def get_db(
    state: Annotated[AppState, Depends(get_app_state)],
) -> AsyncIOMotorDatabase:
    return state.db_client[state.settings.mongodb_db]


def get_note_repo(
    db: Annotated[AsyncIOMotorDatabase, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NoteRepository:
    return NoteRepository(db, collection=settings.mongodb_collection)


def get_transcribe_service(
    state: Annotated[AppState, Depends(get_app_state)],
    note_repo: Annotated[NoteRepository, Depends(get_note_repo)],
    progress_store: Annotated[ProgressStore, Depends(get_progress_store)],
) -> TranscribeService:
    settings = state.settings
    if settings.llm_provider == "anthropic":
        llm_api_key = settings.anthropic_api_key
    elif settings.llm_provider == "openai":
        llm_api_key = settings.openai_api_key
    else:  # local or any OpenAI-compatible server
        llm_api_key = ""
    pipeline = PipelineRunner(
        audio_loader=AudioLoaderStep(),
        vad=VADStep(state.vad_model, state.vad_utils, device=settings.device, chunk_max_seconds=settings.vad_chunk_max_seconds),
        diarization=DiarizationStep(
            state.diarization_pipeline,
            embedding_model=state.embedding_model,
            device=settings.device,
            similarity_threshold=settings.diarization_similarity_threshold,
        ),
        verification=VerificationStep(state.embedding_model, device=settings.device),
        stt=STTStep(state.whisper_model, min_duration_seconds=settings.stt_min_duration_seconds),
        llm_norm=LLMNormStep(
            settings.llm_provider,
            settings.llm_model,
            llm_api_key,
            base_url=settings.llm_base_url,
        ),
        progress_store=progress_store,
    )
    return TranscribeService(note_repo, pipeline, progress_store)
