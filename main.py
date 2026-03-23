# entry point
# initializes FastAPI, connects to ChromaDB and uses "lifespan" context manager to start the orchestrator's clock
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core import orchestrator
from memory.archive import initialize_chroma_client
from api.routes_ws import ws_router
from api.routes_http import http_router
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    #STARTUP
    # TODO: connect to chromaDB instance
    print("[FastAPI] Starting lifespan.")
    print("[ChromaDB] Initializing client...")
    app.state.chroma_client = await initialize_chroma_client()
    print("[CLOCK] Starting orchestrator clock...")
    clock_task =  asyncio.create_task(orchestrator.run_clock(interval=config.CLOCK_INTERVAL))

    yield

    #SHUTDOWN
    # TODO: add chromaDB closing handling
    print("[FastAPI] Ending lifespan.")
    print("[CLOCK] Stopping orchestrator clock.")
    clock_task.cancel()

app = FastAPI(lifespan=lifespan)

app.include_router(ws_router)
app.include_router(http_router)
