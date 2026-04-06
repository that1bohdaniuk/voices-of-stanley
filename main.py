# entry point
# initializes FastAPI, connects to ChromaDB and uses "lifespan" context manager to start the orchestrator's clock
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

import core.orchestrator_service
from memory.archive import initialize_chroma_client
import api.routes_ws as routes_ws
import api.routes_http as routes_http
from llm import client

@asynccontextmanager
async def lifespan(app: FastAPI):
    #STARTUP
    print("[FastAPI] Starting lifespan...")

    print("[Ollama] Initializing Ollama client...")
    await client.check_ollama_server()

    print("[ChromaDB] Initializing client...")
    app.state.chroma_client = await initialize_chroma_client()

    print("[Orchestration] Initializing orchestrator service...]")
    app.state.orchestrator = core.orchestrator_service.OrchestratorService()
    await app.state.orchestrator.start()
    try:
        yield
    finally:
        await app.state.orchestrator.stop()

    #SHUTDOWN
    print("[FastAPI] Ending lifespan...")

    print("[Orchestration] Stopping orchestrator service...")
    await app.state.orchestrator.stop()
    app.state.orchestrator = None

    print("[Ollama] Stopping Ollama client...")
    await client.stop_ollama_server()

app = FastAPI(lifespan=lifespan)

app.include_router(routes_ws.ws_router)
app.include_router(routes_http.http_router)
