# Voices of Stanley: AI Agent Guide

## Overview
**Voices of Stanley** is an AI-driven memory and personality system for the game "The Stanley Parable: Ultra Deluxe" mod. It captures in-game events via WebSocket, processes them through an LLM pipeline, and generates dynamic character responses based on an evolving psychoprofile.

### Core Architecture: Event Processing Pipeline
```
C++ Game Mod (WebSocket) → FastAPI Server → Event Buffer → Orchestrator Clock (5s intervals)
    ↓
  Miner (Ollama/Qwen) → Event Chunking & Importance Scoring
    ↓
ChromaDB Vector Store (Time-Weighted RAG with Importance Decay)
    ↓
Pruner (LLM) → Psychoprofile Consolidation (every 5 days in-game)
    ↓
Director → Game Response Payload → Back to Game via WebSocket
```

## Key Services & Entry Points

### 1. **FastAPI Server** (`main.py`)
- **Entry point**: Initializes with lifespan context manager
- **Startup**: Initializes ChromaDB client and starts `orchestrator.run_clock()`
- **WebSocket endpoint** (`/ws`): Receives raw game deltas from C++ mod, stores global `active_connection` reference
- **HTTP endpoint** (`/ping`): Alternative event ingestion for testing; appends to buffer
- **Debug endpoint** (`/debug/state`): Browser monitoring of Stanley's metrics (WIP)

### 2. **Orchestrator Clock** (`core/orchestrator.py`)
- **Runs every 5 seconds** (configurable via `config.CLOCK_INTERVAL`)
- **Core job**: Flushes `state_buffer` → passes events to miner (TODO: currently just prints)
- **Tension calculation**: Placeholder for computing `TENSION_SUM` based on accumulated importance
- **Note**: The miner integration is intentionally stubbed; implement `orchestrator.TENSION_SUM` population

### 3. **State Buffer** (`core/state_buffer.py`)
- **Thread-safe queue** using `asyncio.Lock()` to prevent race conditions between WebSocket writes and orchestrator flushes
- **Lifecycle**: Append via `state_buffer.append()` → Flush via `state_buffer.flush()` (clears list)
- **Critical**: Must preserve `GameEventModel` schema validation during append

## Event Data Model & Schemas

### `GameEventModel` (`api/schemas.py`)
Events flowing through the pipeline must conform to this Pydantic schema:
```python
GameEventModel:
  - id: UUID (auto-generated if omitted)
  - label: str (concise, descriptive event summary)
  - timestamp: float (in-game seconds or Unix epoch; drives decay)
  - location: Optional[str] (in-game room/area; default: "Not assigned.")
  - importance: float (0.1–10.0 scale; 0.1=trivial, 10.0=critical)
  - details: Optional[Dict] (key-value metadata; VALUES must be primitives only—no nested objects)
```

**Pattern**: Events are created by the game, validated by FastAPI/Pydantic, and remain immutable once in ChromaDB.

### `EventExtractionModel` (`api/schemas.py`)
Miner output format (Pydantic-enforced):
```python
EventExtractionModel:
  - extracted_events: List[GameEventModel]
```

### `ProfileJSONModel` (`api/schemas.py`)
Psychoprofile delta format used by Pruner:
```python
ProfileJSONModel:
  - deltas: Dict[str, Any] (e.g., {"paranoia": 5, "aggression": 2})
```
Pruner **accumulates** deltas into `data/psychoprofile.json`.

## LLM Pipeline & Ollama Integration

### Miner (`llm/miner.py`)
- **Model**: `qwen3.5:0.8B` (can be overridden; currently temporary)
- **Task**: Semantic chunking of raw buffer events + importance scoring
- **Input**: Flushed list of `GameEventModel` (JSON serialized)
- **Output**: `EventExtractionModel` (Pydantic-validated)
- **Integration point**: Called from orchestrator clock (TODO: implement)
- **Ollama setup**: Uses `AsyncClient()` from `ollama` library; expects local Ollama service running

### Pruner (`llm/pruner.py`)
- **Model**: `qwen3.5:0.8B` (temp; intended as larger dedicated model)
- **Task**: Convert events into psychoprofile deltas (e.g., behaviors, emotions, traits)
- **Trigger**: Manually called (not yet integrated into orchestrator); should run every 5 in-game days
- **Flow**:
  1. Fetch old events from ChromaDB via `get_all_to_prune_events()` (older than 5 days)
  2. Read current `data/psychoprofile.json` (or start with `{"deltas": {}}`)
  3. LLM generates delta updates
  4. Accumulate: `new_profile[key] = current_profile.get(key, 0) + delta[key]`
  5. Write updated profile back to JSON
  6. Delete pruned events from ChromaDB
- **Error handling**: Falls back to empty profile if file missing; skips update if LLM validation fails

### Director (`llm/director.py`)
- **Status**: Stub (2 lines); not yet implemented
- **Future role**: Ingest psychoprofile + current context → generate in-game response JSON
- **Output destination**: `send_action_to_game()` in `routes_ws.py`

## Memory & Vector Store (ChromaDB)

### Archive (`memory/archive.py`)
- **Wrapper** around ChromaDB's PersistentClient
- **Collection**: "events-collection" with cosine similarity metric
- **Key functions**:

#### Embedding
- `embed()` / `embed_bunch()`: Add events to ChromaDB with metadata (timestamp, importance, location, details)
- **Metadata casting**: Converts `details` dict keys to strings for ChromaDB compatibility

#### Time-Weighted RAG (TWRAG)
- `twrag(_chroma_results, broadness)`: Custom scoring combining cosine similarity + time decay + importance
- **Formula**: `score = cosine_similarity × exp(-(decay_rate / importance) × age_seconds)`
- **Config params**: `TWRAG_DECAY_RATE` (default: 0.1), `CURRENT_TIME`
- **Returns**: Top N results sorted by score (default broadness=5)

#### Retrieval & Purging
- `retrieve(_event)`: Queries ChromaDB for similar events, applies TWRAG, returns top 5
- `purge_events()`: Deletes events older than `EVENT_PURGE_TIME` (default: 60 min) with importance ≤ `EVENT_PURGE_IMPORTANCE_THRESHOLD` (default: 3.5)
- `get_all_to_prune_events()`: Fetches events older than 5 in-game days for pruner consolidation
- `delete_events_by_id()`: Removes events after pruner processes them

### Database Storage
- **ChromaDB file**: `data/chroma.sqlite3` (persistent local storage)
- **Psychoprofile file**: `data/psychoprofile.json` (manually updated by pruner)
- **Note**: ChromaDB uses Hnsw vector index; cosine space configured at collection init

## Configuration (`config.py`)
**All tunable parameters in one file. Critical for debugging:**
- `CLOCK_INTERVAL`: 5s (orchestrator flush frequency)
- `TENSION_THRESHOLD`: 100 (placeholder for game tension calculation)
- `TWRAG_DECAY_RATE`: 0.1 (memory decay exponential factor)
- `EVENT_PURGE_TIME`: 60 minutes (age threshold for trivial events)
- `EVENT_PURGE_IMPORTANCE_THRESHOLD`: 3.5 (importance floor for purging)
- `EVENT_PRUNE_TIME_THRESHOLD_SECONDS`: 5 days (consolidation window)
- `CURRENT_TIME`: `time.time()` (updated at startup; consider refreshing in clock loop)

## Testing & Development Workflows

### Test Utilities (`misc/`)
- **`test_events_gen.py`**: Generates synthetic `GameEventModel` fixtures (10 random days ago, varied importance)
- **`test_llm.py`**: Mocks Ollama responses; tests miner/pruner without LLM service
  - Uses `@patch` + `AsyncMock` for isolation
  - Example: Test miner's buffer flush + JSON parsing; verify pruner's profile accumulation math
- **`test_routes_and_buffer.py`**: Tests WebSocket/HTTP endpoints + buffer concurrency

### Running Tests
```bash
pytest misc/test_llm.py -v                    # Test LLM mocks
pytest misc/ -v                                # All misc tests
pytest tests/test_chroma.py -v                 # ChromaDB integration tests
pytest --asyncio-mode=auto -v                  # Run all with async support
```

### Local Development
```bash
# Start Ollama service (if using local inference)
ollama serve

# In another terminal, run the FastAPI server
uvicorn main:app --reload

# Test WebSocket connection (from another terminal or client)
websocat ws://localhost:8000/ws
# Send: {"label": "Test event", "timestamp": 1234567890, "importance": 5.0}
```

## Project-Specific Patterns & Conventions

### 1. **Async-First Design**
- All I/O operations use `async/await` (WebSocket, ChromaDB queries, LLM calls)
- CPU-bound sync operations (ChromaDB, vector embeddings) delegated to threads via `asyncio.to_thread()`
- **Pattern**: `await asyncio.to_thread(_collection.query, ...)` prevents GIL blocking

### 2. **Global State Management**
- `active_connection` (WebSocket): Global reference maintained in `routes_ws.py` for Director → Game messaging
- `_collection` (ChromaDB): Global collection reference in `archive.py`
- `_buffer` (list): Global state buffer in `state_buffer.py` protected by `asyncio.Lock()`
- **Convention**: Use module-level globals for singleton services; protect mutable globals with locks

### 3. **Pydantic Schemas as Contracts**
- All data flowing between services validated by Pydantic models
- **Input validation**: FastAPI automatically validates incoming JSON via schema
- **Output validation**: Miner/Pruner outputs validated before use (e.g., `EventExtractionModel.model_validate_json()`)
- **Pattern**: Use `Field(description=...)` for inline documentation; self-documenting schemas

### 4. **Error Handling: Silent Failures with Logging**
- LLM calls wrapped in try/except; failures log via `logging` module but don't halt pipeline
- ChromaDB operations similarly graceful (e.g., pruner skips update if validation fails)
- **Philosophy**: Degradation over crash; always keep game responsive

### 5. **LLM Integration via Ollama**
- Local Ollama service assumed running (`ollama serve`)
- Models are swappable (currently using placeholder `qwen3.5:0.8B`)
- **Pattern**: Use `AsyncClient()` for non-blocking calls; format structured outputs with `format=Model.model_json_schema()`
- **JSON Schema Enforcement**: Models return Pydantic-validated JSON; LLM instructed to follow schema

### 6. **Event Metadata as Extensible Details**
- `GameEventModel.details` is a catch-all for game-specific context (inventory, health, etc.)
- **Constraint**: Values must be primitives (str, int, float, bool)—no nested dicts
- **Pattern**: Store all extra data here; Miner/Director extract as needed from flattened structure

### 7. **Time-Based Decay in Retrieval**
- Older, low-importance events fade from prominence automatically
- **TWRAG formula** combines semantic similarity + temporal relevance
- **Config-driven tuning**: Adjust `TWRAG_DECAY_RATE` to control memory "freshness"

## Common Development Tasks

### Adding a New Event Type
1. Update `GameEventModel.details` schema to document new keys
2. Game mod sends event via WebSocket with new detail field
3. Miner's system prompt auto-processes (no code change needed if Ollama handles it)

### Debugging Event Flow
1. Check `/debug/state` HTTP endpoint
2. Watch console output from orchestrator clock (`[CLOCK]` prefix)
3. Query ChromaDB directly: `data/chroma.sqlite3` (SQLite; use SQLite browser)
4. Inspect `data/psychoprofile.json` for pruner output

### Integrating New LLM Model
1. Update `miner.py` and `pruner.py`: change `model="..."` parameter
2. Test with `misc/test_llm.py` to verify output schema compliance
3. Adjust system prompts as needed (stored in `llm/miner-system` and `llm/pruner-system.txt`)

### Tuning Memory Behavior
1. Adjust `config.py` parameters: `CLOCK_INTERVAL`, `TWRAG_DECAY_RATE`, `EVENT_PURGE_TIME`
2. Re-run `misc/test_events_gen.py` to generate fresh test fixtures
3. Validate via `retrieve()` calls and checking returned scores

## External Dependencies & Integrations
- **ChromaDB** 1.5.5: Vector DB with Hnsw indexing
- **FastAPI** 0.135.1 + **Uvicorn** 0.42.0: HTTP/WebSocket server
- **Ollama** 0.6.1: Local LLM inference (must be running separately)
- **Pydantic** 2.12.5: Data validation & serialization
- **OpenTelemetry** (1.40.0): Observability (setup but not actively used yet)

## Known TODOs & Gaps
1. **Orchestrator → Miner integration**: Miner call placeholder; implement `orchestrator.TENSION_SUM` calculation
2. **Director implementation**: Stub only; needs system prompt + integration with `active_connection`
3. **Profile Manager**: (`memory/profile_manager.py`) Not yet implemented; should wrap psychoprofile I/O
4. **Real-time `/debug/state` metrics**: Currently returns empty `{"status": "Listening"}`
5. **Pruner scheduling**: Not integrated into orchestrator; must be called manually or via cron

## References
- **Architecture diagram**: https://miro.com/app/board/uXjVG1GCamU=/ (mentioned in README)
- **Test fixtures**: `misc/test_events_gen.py` (synthetic event generation)
- **System prompts**: `llm/miner-system` and `llm/pruner-system.txt`

