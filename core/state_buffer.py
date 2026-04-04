# thread-safe, temporary holding area for the validated game deltas before orchestrator passes them to the miner
# asyncio.Lock() to ensure that the WebSocket listener doesn't write to the list at the exact millisecond the Orchestrator is trying to empty it.
import asyncio
from typing import Optional

from api import schemas

_buffer: list[schemas.GameEventModel] = []
_lock = asyncio.Lock()

async def append(event: schemas.GameEventModel):
    #called in routes_ws
    async with _lock:
        _buffer.append(event)

async def flush() -> list[schemas.GameEventModel]:
    # flushed by orchestrator clock and sends buffer to miner
    async with _lock:
        temp: list[schemas.GameEventModel] = _buffer.copy()
        _buffer.clear()
        return temp

async def get_last_event_before(cutoff_ts: float) -> Optional[schemas.GameEventModel]:
    async with _lock:
        # newest -> oldest
        for event in reversed(_buffer):
            if event.timestamp <= cutoff_ts:
                return event
        return None


async def get_last_event() -> Optional[schemas.GameEventModel]:
    async with _lock:
        if not _buffer:
            return None
        return _buffer[-1]
