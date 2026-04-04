# entry point
# initializes FastAPI, connects to ChromaDB and uses "lifespan" context manager to start the orchestrator's clock
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core import orchestrator_service
from memory.archive import initialize_chroma_client
from api.routes_ws import ws_router
from api.routes_http import http_router
from llm import client
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    #STARTUP
    print("[FastAPI] Starting lifespan...")

    print("[Ollama] Initializing Ollama client...")
    await client.check_ollama_server()

    print("[ChromaDB] Initializing client...")
    app.state.chroma_client = await initialize_chroma_client()

    print("[CLOCK] Starting orchestrator clock...")
    app.state.orchestrator = orchestrator_service.OrchestratorService()
    await app.state.orchestrator.start()
    try:
        yield
    finally:
        await app.state.orchestrator.stop()

    #SHUTDOWN
    print("[FastAPI] Ending lifespan...")

    print("[CLOCK] Stopping orchestrator clock...")

    print("[Ollama] Stopping Ollama client...")
    await client.stop_ollama_server()

app = FastAPI(lifespan=lifespan)

app.include_router(ws_router)
app.include_router(http_router)
