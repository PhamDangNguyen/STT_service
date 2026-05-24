# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Speech-to-Text microservice. Receives a `note_id`, fetches audio from MongoDB, runs a full ML pipeline, and writes the structured transcript back to the note document.

Pipeline:
```
AudioLoaderStep → VADStep → DiarizationStep → VerificationStep → STTStep → LLMNormStep
     10%            20%          40%               50%            50–85%       100%
```

## Commands

```bash
uv sync                      # Install dependencies
uv run main.py               # Start on :8000
uv run test/mongoDB/test.py  # Run MongoDB tests
uv add <package>             # Add a dependency
```

Test the API manually:
```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"note_id":"<id>"}'

curl "http://localhost:8000/api/transcribe/<id>/status"
```

## Architecture

```
app/
├── api/           # Route handlers (routes.py), request/response schemas, Depends helpers
├── pipeline/      # PipelineStep[I,O] base class + 6 step implementations + models.py
├── repositories/  # NoteRepository — all MongoDB I/O (motor)
├── services/      # TranscribeService — orchestrates PipelineRunner as background task
├── stores/        # InMemoryProgressStore; swap with Redis via Strategy pattern
├── config.py      # Settings loader: .env (secrets) merged with config.json
├── state.py       # AppState dataclass — holds all loaded ML models
└── exceptions.py  # Domain exceptions
main.py            # FastAPI app + lifespan (model loading at startup)
config.json        # Non-sensitive settings (models, device, activate flags, TTL)
```

**Key design rules:**
- Route handlers contain no business logic — only validate + delegate to service layer.
- All MongoDB access goes through `NoteRepository`. Never call `motor` directly from services or pipeline steps.
- ML inference runs in `ThreadPoolExecutor` via `asyncio.run_in_executor`; DB/network calls are async.
- All cross-module function signatures use Pydantic models (never raw `dict`).

## Configuration

**`config.json`** — safe to commit. Edit here to change models, device, and activate flags.

**`.env`** — secrets only:
```env
MONGODB_USERNAME=
MONGDB_PASSWORD=        # intentional typo — one "O", matches test fixtures
PYANNOTE_TOKEN=hf_...
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Activate flags (important for local dev without GPU)

Each ML component can be disabled individually in `config.json`:
```json
{
  "vad":         { "activate": false },
  "diarization": { "activate": false },
  "embedding":   { "activate": false },
  "stt":         { "activate": false },
  "llm":         { "activate": false }
}
```
When disabled, the model is skipped at startup and the step becomes a passthrough.

### Common configuration changes

| Goal | Where |
|---|---|
| Switch to CPU | `config.json` → `"device": "cpu"`, `stt.compute_type: "int8"` |
| Use self-hosted LLM | `config.json` → `llm.base_url: "http://..."`, `llm.model: "<model-name>"` |
| Switch to Anthropic | `config.json` → `llm.provider: "anthropic"`, then set `ANTHROPIC_API_KEY` in `.env` |
| Tune diarization accuracy | `config.json` → `diarization.similarity_threshold` (default `0.9`; lower = merge more aggressively) |
| Use Redis for ProgressStore | `.env` → `REDIS_URL=redis://...` |

## MongoDB

- **Database:** `funny_hunter` | **Collection:** `notes`
- `NoteRepository.get_by_id` returns a typed Pydantic model, not a raw dict.
- `NoteRepository.update_transcript` updates only transcript fields and `updatedAt`.
- Missing note → `NoteNotFoundError`.

## Pipeline Data Flow

Each step is typed via Pydantic models in `app/pipeline/models.py`:

```
AudioLoadInput → AudioLoadOutput (float32, 16kHz mono PCM)
VADInput       → VADOutput       (list[AudioChunk] with timestamps)
DiarizationInput → DiarizationOutput  (list[DiarizedSegment] with global speaker labels)
VerificationInput → VerificationOutput (list[VerifiedSegment] with named speakers + audio)
STTInput       → STTOutput       (list[TranscribedSegment] with text)
LLMNormInput   → LLMNormOutput   (list[TranscribedSegment] with corrected text)
```

`STTStep` receives a `progress_callback: Callable[[chunk_index, total_chunks], Awaitable[None]]` injected by `PipelineRunner` — STTStep does not depend on ProgressStore directly.

## Diarization — cross-chunk speaker identity

`DiarizationStep` runs pyannote per VAD chunk (which only assigns local labels). It maintains an in-memory **global registry** for the duration of one request:
- Each local speaker label → embedding (pyannote/embedding)
- Embedding compared via cosine similarity against registry
- `sim >= similarity_threshold` → reuse existing global speaker, update embedding with running mean
- `sim < threshold` → new global speaker (`SPEAKER_00`, `SPEAKER_01`, …)

`VerificationStep` then optionally renames global labels to real names using pre-stored `speaker_profiles` embeddings.

## Domain Exceptions

`NoteNotFoundError`, `AudioFetchError`, `UnsupportedAudioFormatError`, `PipelineStepError(step_name, cause)`, `ProgressNotFoundError`
