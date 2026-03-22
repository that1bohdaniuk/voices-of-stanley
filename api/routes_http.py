# debug dashboard
# has simple GET methods that enable the browser monitoring of stanley metrics
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter

from api.schemas import GameEvent
from core import state_buffer

# import internal state variables here later
# from core.orchestrator import tension_sum, idle_counter

http_router = APIRouter()

@http_router.get("/")
async def root():
    return {"message": "Stanley API is running."}

@http_router.get("/debug/state")
async def get_state():
    """Open this in your browser to monitor Stanley's brain."""
    return {
        "status": "Listening",
        # "tension": tension_sum,
        # "idle_count": idle_counter
    }

@http_router.post("/ping")
async def ping(game_event: GameEvent):
    print(f"Event Ping received. {game_event.id}.\nLabel is {game_event.label}")
    await state_buffer.append(game_event)

