from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import load_settings
from app.logging_setup import setup_logging, get_logger
from app.state import AppState
from app.stores.memory_store import InMemoryProgressStore
from app.api.routes import router
from rich import print

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    setup_logging(settings)

    logger.info("Starting STT Service (device=%s)", settings.device)

    db_client = AsyncIOMotorClient(
        f"mongodb+srv://{settings.mongodb_username}:{settings.mongdb_password}"
        f"@cluster0.0h1sfbi.mongodb.net/{settings.mongodb_db}"
    )
    print("GetSettings: ", settings)
    progress_store = InMemoryProgressStore(ttl_seconds=settings.progress_ttl_seconds)

    vad_model = None
    vad_utils = None
    diarization_pipeline = None
    embedding_model = None
    whisper_model = None

    if settings.vad_activate:
        try:
            import torch
            logger.info("Loading VAD model (silero-vad)...")
            vad_model, vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            vad_model = vad_model.to(settings.device)
            logger.info("VAD model loaded (device=%s).", settings.device)
        except Exception as e:
            logger.warning("VAD model not loaded: %s", e)
    else:
        logger.info("VAD model skipped (activate=false).")

    if settings.diarization_activate:
        try:
            import torch
            from pyannote.audio import Pipeline
            logger.info("Loading diarization pipeline (%s)...", settings.diarization_model)
            diarization_pipeline = Pipeline.from_pretrained(
                settings.diarization_model,
                token=settings.pyannote_token,
            )
            diarization_pipeline = diarization_pipeline.to(torch.device(settings.device))
            logger.info("Diarization pipeline loaded (device=%s).", settings.device)
        except Exception as e:
            logger.warning("Diarization pipeline not loaded: %s", e)
    else:
        logger.info("Diarization pipeline skipped (activate=false).")

    if settings.embedding_activate:
        try:
            import torch
            from pyannote.audio import Model, Inference
            logger.info("Loading embedding model (%s)...", settings.embedding_model)
            _raw_embed = Model.from_pretrained(
                settings.embedding_model,
                token=settings.pyannote_token,
            )
            _raw_embed = _raw_embed.to(torch.device(settings.device))
            _raw_embed.eval()
            embedding_model = Inference(_raw_embed, window="whole")
            logger.info("Embedding model loaded (device=%s).", settings.device)
        except Exception as e:
            logger.warning("Embedding model not loaded: %s", e)
    else:
        logger.info("Embedding model skipped (activate=false).")

    if settings.stt_activate:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading Whisper model (%s, compute_type=%s)...",
                settings.stt_model,
                settings.stt_compute_type,
            )
            whisper_model = WhisperModel(
                settings.stt_model,
                device=settings.device,
                compute_type=settings.stt_compute_type,
            )
            logger.info("Whisper model loaded.")
        except Exception as e:
            logger.warning("Whisper model not loaded: %s", e)
    else:
        logger.info("Whisper model skipped (activate=false).")

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

    logger.info("All models loaded. STT Service is ready.")
    yield

    db_client.close()
    logger.info("STT Service shut down.")


app = FastAPI(title="STT Service", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
