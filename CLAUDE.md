# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This project is a Speech-to-Text microservice. It receives a `note_id`, fetches the corresponding audio metadata from MongoDB, runs the full audio-processing pipeline, and writes the structured transcript back to the same MongoDB note document.

The pipeline is:

```text
VAD → Diarization → Speaker Verification → STT → LLM Normalization
```

The service is still in early development. At the moment, only a stub `main.py` exists. The architecture described below is the intended implementation. See also `CLAUDE_1.md` for the original design specification.

## Commands

This project uses `uv` for package management. Do not use `pip` directly.

```bash
uv sync                     # Install dependencies
uv run main.py              # Run the application
uv run test/mongoDB/test.py # Run a specific MongoDB test file
uv add <package>            # Add a dependency
```

## Architecture

```text
POST /api/transcribe
       │
       ├── ProgressStore.init(note_id)           ← Set status=pending, progress=0
       │
TranscribeService                                 (runs as a background task)
       ├── NoteRepository.get_by_id(note_id)     ← Fetch note document from MongoDB
       ├── PipelineRunner.run(PipelineInput)
       │        ├── 1. AudioLoaderStep  — Download + decode to 16 kHz mono PCM (httpx + ffmpeg)
       │        │        └── progress: 10%
       │        ├── 2. VADStep          — Remove silence (silero-vad)
       │        │        └── progress: 20%
       │        ├── 3. DiarizationStep  — Speaker segmentation (pyannote/speaker-diarization-3.1)
       │        │        └── progress: 40%  ← hard milestone after diarization
       │        ├── 4. VerificationStep — Match segments to named speaker profiles (cosine similarity)
       │        │        └── progress: 50%
       │        ├── 5. STTStep          — Transcribe each segment (faster-whisper large-v3)
       │        │        └── progress: 50% + round(chunk_index / total_chunks × 35%) → up to 85%
       │        └── 6. LLMNormStep      — Vietnamese spelling and punctuation normalization
       │                 └── progress: 100%, status=done
       └── NoteRepository.update_transcript(note_id, segments) ← Persist transcript to MongoDB

GET /api/transcribe/{note_id}/status
       └── ProgressStore.get(note_id)            ← Return ProgressResponse
```

Heavy ML models such as Whisper, pyannote, and speaker-embedding models are loaded once during FastAPI startup through the `lifespan` context. They are stored in a single `AppState` dataclass and injected with `Depends`.

Route handlers must contain no business logic. They should only validate requests and delegate work to the service layer.

## MongoDB Schema

The service stores and updates note documents in the following MongoDB collection:

```text
Database:   funny_hunter
Collection: notes
```

Example document:

```json
{
  "_id": "<note_id>",
  "audio_url": "http://...",
  "transcript": [
    {
      "start": 3.72,
      "duration": 2.52,
      "text": "Ừ.",
      "speaker": "Minh Nguyễn Lê"
    }
  ],
  "summary": "",
  "mindmap": "",
  "slide_url": "",
  "suggestions": [],
  "createdAt": "...",
  "updatedAt": "..."
}
```

### MongoDB Access Rules

- All MongoDB access must go through `NoteRepository`.
- Services and pipeline steps must not call `pymongo` or `motor` directly.
- Use `_id` as the canonical note identifier.
- `NoteRepository.get_by_id(note_id)` must return a typed Pydantic model, not a raw MongoDB dictionary.
- `NoteRepository.update_transcript(note_id, segments)` must update only the transcript-related fields and `updatedAt`.
- If no document exists for the given `note_id`, raise `NoteNotFoundError`.

## Progress Tracking

### ProgressStore

`ProgressStore` stores the processing state for each `note_id`. The default implementation may use an in-memory dictionary, which is sufficient for a single-process deployment.

If the service needs to scale to multiple workers or multiple instances, replace the in-memory implementation with Redis. This should be done through the Strategy pattern by injecting the desired `ProgressStore` implementation through the constructor.

```python
class ProgressStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"

class ProgressRecord(BaseModel):
    note_id:      str
    status:       ProgressStatus
    progress:     int            # 0–100
    current_step: str | None = None
    error:        str | None = None
    updated_at:   datetime

class ProgressStore(Protocol):
    async def init(self, note_id: str) -> None: ...
    async def update(self, note_id: str, progress: int, step: str) -> None: ...
    async def finish(self, note_id: str) -> None: ...
    async def fail(self, note_id: str, error: str) -> None: ...
    async def get(self, note_id: str) -> ProgressRecord | None: ...
```

### Progress Allocation

| Step | Update condition | Progress after step |
|---|---|---|
| `AudioLoaderStep` | Completed | 10% |
| `VADStep` | Completed | 20% |
| `DiarizationStep` | Completed | **40%** hard milestone |
| `VerificationStep` | Completed | 50% |
| `STTStep` | After each processed chunk `i` out of `total` chunks | `50 + round(i / total * 35)` % → maximum 85% |
| `LLMNormStep` | Completed | 100%, status → `done` |

`STTStep` must call `progress_store.update()` after each chunk, not only after the full STT step is complete. This allows the client to observe incremental progress from 50% to 85%.

Recommended safe formula:

```python
progress = 50 + round(processed_chunks / total_chunks * 35)
progress = min(progress, 85)
```

Where `processed_chunks` is the number of chunks already completed. If `chunk_index` starts at `0`, use:

```python
processed_chunks = chunk_index + 1
```

### GET `/api/transcribe/{note_id}/status`

```json
Response 200:
{
  "note_id": "abc123",
  "status": "processing",
  "progress": 62,
  "current_step": "STTStep",
  "error": null
}
```

Valid status values:

```text
pending | processing | done | error
```

```text
Response 404:
note_id does not exist in ProgressStore, usually because POST /api/transcribe has not been called yet.
```

Response requirements:

- Set header: `Cache-Control: no-store`.
- The client should poll this endpoint every 1–2 seconds.
- When `status == "done"` or `status == "error"`, the client should stop polling.
- `ProgressRecord` should remain in the store for at least 1 hour after `done` or `error` using TTL.

## Implementation Notes

- `POST /api/transcribe` must immediately return `202 Accepted` without blocking.
- The response body should include:

```json
{
  "note_id": "...",
  "status_url": "/api/transcribe/{note_id}/status"
}
```

- The pipeline should run in `BackgroundTasks` from FastAPI or through `asyncio.create_task`.
- `ProgressStore` must be injected into `TranscribeService`.
- `ProgressStore` must also be injected into the `GET /status` route handler via `Depends`.
- If any pipeline step raises an exception, `TranscribeService` must catch it and call:

```python
await progress_store.fail(note_id, str(error))
```

## Design Patterns

- **Pipeline pattern** — each processing stage is a `PipelineStep[I, O]` abstract base class with typed Pydantic input/output models. `PipelineRunner` chains the steps and owns logging/timing.
- **Repository pattern** — all MongoDB I/O goes through `NoteRepository`. Services and pipeline steps never access MongoDB directly.
- **Strategy pattern** — STT, diarization, speaker verification backends, and `ProgressStore` are injected through constructors so they can be replaced without changing pipeline logic.
- **Async I/O, sync ML** — database and network calls are asynchronous; ML inference runs inside a `ThreadPoolExecutor` through `asyncio.run_in_executor`.

## Implementation Rules

- All cross-module function signatures must use Pydantic models.
- Do not pass raw `dict` objects between services or pipeline steps.
- `LLMNormStep` should normalize Vietnamese spelling, punctuation, and formatting while preserving the original meaning.
- If using Anthropic, `LLMNormStep` must use `cache_control: "ephemeral"` on the system prompt to avoid re-tokenizing the same prompt on every request.
- Use `type AudioArray = np.ndarray` on Python 3.12+ or `TypeAlias` for all `np.ndarray` usages.
- `STTStep` must accept:

```python
progress_callback: Callable[[int, int], Awaitable[None]]
```

Where the arguments are:

```text
chunk_index, total_chunks
```

This allows `PipelineRunner` to inject progress tracking without making `STTStep` depend directly on `ProgressStore`.

## Configuration

Settings are split into two files:

- `config.json` — non-sensitive values. Safe to commit to version control.
- `.env` — secrets only (API keys, database credentials). Never commit this file.

Both are loaded at startup and merged into a single `Settings` object via `pydantic-settings`.

### `config.json`

Edit this file to change model choices, device settings, and tunable parameters. No environment variables need to be touched for these changes.

```json
{
  "mongodb": {
    "db": "funny_hunter",
    "collection": "notes"
  },
  "vad": {
    "model": "silero-vad"
  },
  "diarization": {
    "model": "pyannote/speaker-diarization-3.1"
  },
  "embedding": {
    "model": "pyannote/embedding"
  },
  "stt": {
    "model": "large-v3",
    "compute_type": "float16"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "device": "cuda",
  "progress": {
    "ttl_seconds": 3600
  }
}
```

### `.env` (secrets only)

Only credentials and API keys belong here. Do not add model names, device settings, or other non-sensitive values to `.env`.

```env
# MongoDB credentials
MONGODB_USERNAME=
MONGDB_PASSWORD=        # Note: one "O" — matches the typo in test/mongoDB/test.py

# Hugging Face token (required for gated pyannote models)
PYANNOTE_TOKEN=hf_...

# LLM provider — set whichever matches llm.provider in config.json
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional: uncomment to use Redis instead of in-memory ProgressStore
# REDIS_URL=redis://localhost:6379
```

### Settings loader

`pydantic-settings` loads `.env` for secrets. `config.json` is loaded separately and passed in to populate the remaining fields. The result is a single `Settings` instance injected via `Depends` throughout the app.

```python
class Settings(BaseSettings):
    # Secrets — from .env only
    mongodb_username:  str
    mongdb_password:   str        # keep the typo to match existing test fixtures
    pyannote_token:    str
    openai_api_key:    str = ""
    anthropic_api_key: str = ""
    redis_url:         str = ""

    # Non-sensitive — from config.json (with safe defaults)
    mongodb_db:           str = "funny_hunter"
    mongodb_collection:   str = "notes"
    vad_model:            str = "silero-vad"
    diarization_model:    str = "pyannote/speaker-diarization-3.1"
    embedding_model:      str = "pyannote/embedding"
    stt_model:            str = "large-v3"
    stt_compute_type:     str = "float16"
    llm_provider:         str = "openai"
    llm_model:            str = "gpt-4o"
    device:               str = "cuda"
    progress_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

### Common configuration changes

| What you want to change | Where to edit |
|---|---|
| Switch LLM provider or model | `config.json` → `llm.provider`, `llm.model` |
| Switch to CPU inference | `config.json` → `"device": "cpu"`, `stt.compute_type": "int8"` |
| Change diarization or STT model | `config.json` → `diarization.model`, `stt.model` |
| Use Redis for ProgressStore | `.env` → uncomment `REDIS_URL` |
| Rotate API keys | `.env` only |
| Change MongoDB database name | `config.json` → `mongodb.db` |

### Notes

- To switch from OpenAI to Anthropic: set `llm.provider = "anthropic"` and `llm.model = "claude-sonnet-4-5"` in `config.json`, then set `ANTHROPIC_API_KEY` in `.env`.
- Keep `MONGDB_PASSWORD` (one O) unless all test fixtures are updated in the same migration.

## Domain Exceptions

- `NoteNotFoundError` — the given `note_id` does not exist in MongoDB.
- `AudioFetchError` — the service cannot download the file from `audio_url`.
- `UnsupportedAudioFormatError` — the audio extension is not one of `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.aac`, or `.opus`.
- `PipelineStepError(step_name, cause)` — wraps any exception raised by a pipeline step.
- `ProgressNotFoundError` — the given `note_id` does not exist in `ProgressStore`, usually when calling the status endpoint for an unknown job.
