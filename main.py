# entry point
# initializes FastAPI, connects to ChromaDB and uses "lifespan" context manager to start the orchestrator's clock
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
import chromadb

from core import orchestrator
from api.routes_ws import ws_router
from api.routes_http import http_router
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    #STARTUP
    # TODO: connect to chromaDB instance
    print("[FastAPI] Starting lifespan.")
    print("[ChromaDB] Initializing client...")
    chroma_client = chromadb.PersistentClient(path="./data/")
    print("[CLOCK] Starting orchestrator clock.")
    clock_task =  asyncio.create_task(orchestrator.run_clock(interval=config.CLOCK_INTERVAL))
    #app.state.vector_db = init_chroma_client()

    yield

    #SHUTDOWN
    # TODO: add chromaDB closing handling
    print("[FastAPI] Ending lifespan.")
    print("[CLOCK] Stopping orchestrator clock.")
    clock_task.cancel()
    # app.state.vector_db.close()

app = FastAPI(lifespan=lifespan)

app.include_router(ws_router)
app.include_router(http_router)
