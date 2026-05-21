import json
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Secrets — from .env only
    mongodb_username: str
    mongdb_password: str        # keep the typo to match existing fixtures
    pyannote_token: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    redis_url: str = ""

    # Non-sensitive — from config.json (with safe defaults)
    mongodb_db: str = "funny_hunter"
    mongodb_collection: str = "notes"
    vad_model: str = "silero-vad"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/embedding"
    stt_model: str = "large-v3"
    stt_compute_type: str = "float16"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    device: str = "cuda"
    progress_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "config.json"
    overrides: dict = {}
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        mapping = {
            ("mongodb", "db"): "mongodb_db",
            ("mongodb", "collection"): "mongodb_collection",
            ("vad", "model"): "vad_model",
            ("diarization", "model"): "diarization_model",
            ("embedding", "model"): "embedding_model",
            ("stt", "model"): "stt_model",
            ("stt", "compute_type"): "stt_compute_type",
            ("llm", "provider"): "llm_provider",
            ("llm", "model"): "llm_model",
        }
        for (section, key), field in mapping.items():
            if section in raw and key in raw[section]:
                overrides[field] = raw[section][key]
        if "device" in raw:
            overrides["device"] = raw["device"]
        if "progress" in raw and "ttl_seconds" in raw["progress"]:
            overrides["progress_ttl_seconds"] = raw["progress"]["ttl_seconds"]
    return Settings(**overrides)
