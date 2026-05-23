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
    vad_activate: bool = True
    vad_model: str = "silero-vad"
    vad_chunk_max_seconds: float = 300.0
    diarization_activate: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_similarity_threshold: float = 0.75
    embedding_activate: bool = True
    embedding_model: str = "pyannote/embedding"
    stt_activate: bool = True
    stt_model: str = "large-v3"
    stt_compute_type: str = "float16"
    stt_min_duration_seconds: float = 0.5
    llm_activate: bool = True
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_base_url: str = ""
    device: str = "cuda"
    progress_ttl_seconds: int = 3600
    log_level: str = "INFO"
    log_folder: str = "logs"
    log_max_bytes: int = 10485760  # 10 MB
    log_backup_count: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "config.json"
    overrides: dict = {}
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        mapping = {
            ("mongodb", "db"): "mongodb_db",
            ("mongodb", "collection"): "mongodb_collection",
            ("vad", "activate"): "vad_activate",
            ("vad", "model"): "vad_model",
            ("vad", "chunk_max_seconds"): "vad_chunk_max_seconds",
            ("diarization", "activate"): "diarization_activate",
            ("diarization", "model"): "diarization_model",
            ("diarization", "similarity_threshold"): "diarization_similarity_threshold",
            ("embedding", "activate"): "embedding_activate",
            ("embedding", "model"): "embedding_model",
            ("stt", "activate"): "stt_activate",
            ("stt", "model"): "stt_model",
            ("stt", "compute_type"): "stt_compute_type",
            ("stt", "min_duration_seconds"): "stt_min_duration_seconds",
            ("llm", "activate"): "llm_activate",
            ("llm", "provider"): "llm_provider",
            ("llm", "model"): "llm_model",
            ("llm", "base_url"): "llm_base_url",
        }
        for (section, key), field in mapping.items():
            if section in raw and key in raw[section]:
                overrides[field] = raw[section][key]
        if "device" in raw:
            overrides["device"] = raw["device"]
        if "progress" in raw and "ttl_seconds" in raw["progress"]:
            overrides["progress_ttl_seconds"] = raw["progress"]["ttl_seconds"]
        if "logging" in raw:
            lg = raw["logging"]
            if "level" in lg:
                overrides["log_level"] = lg["level"]
            if "folder" in lg:
                overrides["log_folder"] = lg["folder"]
            if "max_bytes" in lg:
                overrides["log_max_bytes"] = lg["max_bytes"]
            if "backup_count" in lg:
                overrides["log_backup_count"] = lg["backup_count"]
    return Settings(**overrides)
