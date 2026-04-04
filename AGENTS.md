# AGENTS Guide

## Project Snapshot
- `voices-of-stanley` is an async FastAPI service that ingests game telemetry, stores memory in ChromaDB, and sends AI director actions back over WebSocket.
- Main flow: game -> `api/routes_ws.py` -> `core/state_buffer.py` -> LLM modules (`llm/*`) -> memory (`memory/archive.py`) -> action back via `api/routes_ws.send_action_to_game`.

## Architecture That Matters
- App lifecycle is in `main.py`: startup checks/starts Ollama, initializes persistent Chroma (`./data/`), and starts `OrchestratorService`; shutdown stops orchestrator then Ollama.
- `core/orchestrator_service.py` owns async coordination (`_reducer_loop`, `_scheduler_loop`, `_worker_loop`) and gates LLM concurrency with `asyncio.Semaphore(2)`.
- `api/schemas.py` is the contract layer; incoming telemetry must match `GameEventModel`, director output must match `DirectorEventModel`.
- `memory/archive.py` is the memory boundary: `embed*`, `retrieve`, `purge_events`, `get_all_to_prune_events`, `delete_events_by_id`.

## Current Data/Control Flows
- WebSocket `/ws` validates payloads and appends events to in-memory buffer (`state_buffer.append`).
- HTTP `/ping` is a lightweight ingest/debug path that also appends to the same buffer.
- Director sends outbound payloads through the module-global `active_connection` in `api/routes_ws.py`.
- Orchestrator job types are `RUN_MINER`, `RUN_DIRECTOR`, `RUN_PRUNER`; prune scheduling runs every 5s and triggers when 5 in-game days elapsed.
- Important: no discovered code path currently publishes ingest signals to `OrchestratorService.publish(...)`, so reducer triggers are not obviously wired from routes yet.

## Project-Specific Conventions
- Keep async boundaries explicit; blocking Chroma calls are offloaded with `asyncio.to_thread(...)` in `memory/archive.py`.
- Event metadata conventions: always include `timestamp` and `importance`; optional `location`; flatten `details` into metadata keys.
- TWRAG scoring is custom: cosine similarity multiplied by exponential decay weighted by importance (`memory/archive.py::twrag`).
- LLM calls are schema-constrained (`format=...model_json_schema()`), then validated with Pydantic before use.
- After each LLM call, model unload is expected in `finally` via `unload_ollama_model(...)`.

## Developer Workflow (Verified)
- Run tests from repo root with import path set:
  - `PYTHONPATH=. pytest tests -q`
- Plain `pytest -q` currently fails module imports (`main`, `api`, `llm`, `memory`) because package path is not auto-set.
- Current known failing test: `tests/test_llm.py::test_pruner` calls `pruner.prune()` but implementation exposes `run_pruner()`.

## Integration/Dependency Notes
- External runtime dependencies: local Ollama server (`ollama serve`) and Chroma persistent storage under `data/chroma.sqlite3` plus collection files.
- `llm/director.py` and `llm/pruner.py` read/write `../data/psychoprofile.json`; this relative path is CWD-sensitive.
- System prompt/model definitions live in `llm/miner-system` and `llm/pruner-system.txt`; runtime model names come from `config.py`.

## Safe Change Strategy For Agents
- When changing event schema, update all three: `api/schemas.py`, routes validation paths, and tests under `tests/`.
- When changing memory ranking or pruning, run `tests/test_chroma.py` first; it encodes expected retrieval/purge behavior.
- When touching WebSocket behavior, validate both inbound (`/ws`) and outbound (`send_action_to_game`) paths via `tests/test_routes_and_buffer.py`.

