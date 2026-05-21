# STT Service

A Speech-to-Text microservice that receives a `note_id`, fetches audio from MongoDB, runs a full ML pipeline, and writes the structured transcript back to the note document.

## Pipeline

```
AudioLoaderStep → VADStep → DiarizationStep → VerificationStep → STTStep → LLMNormStep
     10%            20%          40%               50%            50–85%       100%
```

Each step reports progress to a `ProgressStore`; the client polls `GET /api/transcribe/{note_id}/status`.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/transcribe` | Submit job → `202 Accepted` immediately |
| `GET`  | `/api/transcribe/{note_id}/status` | Poll progress (`pending/processing/done/error`) |

## Project Structure

```
app/
├── api/           # Route handlers, schemas, Depends helpers
├── pipeline/      # PipelineStep base + 6 step implementations
├── repositories/  # NoteRepository — all MongoDB I/O
├── services/      # TranscribeService — orchestrates pipeline
├── stores/        # InMemoryProgressStore (swap with Redis via Strategy pattern)
├── config.py      # Settings loader (pydantic-settings)
├── state.py       # AppState — holds loaded ML models
└── exceptions.py  # Domain exceptions
main.py            # FastAPI app + lifespan model loading
config.json        # Non-sensitive settings (models, device, TTL)
.env               # Secrets only (API keys, DB credentials) — never commit
```

## Configuration

**`config.json`** — model names, device, LLM provider (safe to commit):
```json
{ "stt": { "model": "large-v3" }, "llm": { "provider": "openai", "model": "gpt-4o" }, "device": "cuda" }
```

**`.env`** — secrets only:
```env
MONGODB_USERNAME=...
MONGDB_PASSWORD=...        # one "O" — intentional
PYANNOTE_TOKEN=hf_...
OPENAI_API_KEY=sk-...
```

To switch to Anthropic: set `llm.provider = "anthropic"` in `config.json` and add `ANTHROPIC_API_KEY` to `.env`.

## MongoDB

- **Database:** `funny_hunter` | **Collection:** `notes`
- All access goes through `NoteRepository` — services never call `motor` directly.
- Transcript segments are written back to the note document on completion.

## Development Commands

```bash
uv sync                      # Install dependencies
uv run main.py               # Start the service on :8000
uv run test/mongoDB/test.py  # Run MongoDB tests
uv add <package>             # Add a dependency
```

## Tech Stack

FastAPI · Motor (async MongoDB) · silero-vad · pyannote/speaker-diarization-3.1 · faster-whisper · OpenAI / Anthropic (LLM normalization) · pydantic-settings
