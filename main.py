from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import load_settings
from app.state import AppState
from app.stores.memory_store import InMemoryProgressStore
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    db_client = AsyncIOMotorClient(
        f"mongodb+srv://{settings.mongodb_username}:{settings.mongdb_password}"
        f"@cluster0.0h1sfbi.mongodb.net/{settings.mongodb_db}"
    )
    progress_store = InMemoryProgressStore(ttl_seconds=settings.progress_ttl_seconds)

    vad_model = None
    vad_utils = None
    diarization_pipeline = None
    embedding_model = None
    whisper_model = None

    try:
        import torch
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
    except Exception as e:
        print(f"[warn] VAD model not loaded: {e}")

    try:
        from pyannote.audio import Pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            settings.diarization_model,
            use_auth_token=settings.pyannote_token,
        )
    except Exception as e:
        print(f"[warn] Diarization pipeline not loaded: {e}")

    try:
        from pyannote.audio import Model
        embedding_model = Model.from_pretrained(
            settings.embedding_model,
            use_auth_token=settings.pyannote_token,
        )
    except Exception as e:
        print(f"[warn] Embedding model not loaded: {e}")

    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            settings.stt_model,
            device=settings.device,
            compute_type=settings.stt_compute_type,
        )
    except Exception as e:
        print(f"[warn] Whisper model not loaded: {e}")

    app.state.app_state = AppState(
        settings=settings,
        db_client=db_client,
        progress_store=progress_store,
        vad_model=vad_model,
        vad_utils=vad_utils,
        diarization_pipeline=diarization_pipeline,
        embedding_model=embedding_model,
        whisper_model=whisper_model,
    )

    yield

    db_client.close()


app = FastAPI(title="STT Service", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
