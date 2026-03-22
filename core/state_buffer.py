# thread-safe, temporary holding area for the validated game deltas before orchestrator passes them to the miner
# asyncio.Lock() to ensure that the WebSocket listener doesn't write to the list at the exact millisecond the Orchestrator is trying to empty it.
import asyncio
from api import schemas

_buffer: list[schemas.GameEvent] = []
_lock = asyncio.Lock()

async def append(event: schemas.GameEvent):
    #called in routes_ws
    async with _lock:
        _buffer.append(event)

async def flush() -> list[schemas.GameEvent]:
    # flushed by orchestrator clock and sends buffer to miner
    async with _lock:
        temp: list[schemas.GameEvent] = _buffer.copy()
        _buffer.clear()
        return temp