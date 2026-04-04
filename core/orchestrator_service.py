# the asyncio background loop
# watches the clock, flushes the buffer, calculates Tension Sum, evaluates importance score triggers
# dictates how llm and memory modules interact
import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from api.schemas import GameEventModel
from core import state_buffer
from llm.director import run_director
from llm.pruner import run_pruner
from llm.miner import run_miner

class SignalType(Enum):
    EVENT_INGESTED = auto()
    IDLE_INCREMENT = auto()
    FORCE_DIRECTOR = auto()

class JobType(Enum):
    RUN_MINER = auto()
    RUN_DIRECTOR = auto()
    RUN_PRUNER = auto()

@dataclass
class Signal:
    type: SignalType
    value: float | int
    timestamp: Optional[float] = None

@dataclass
class State:
    tension_sum: float = 0.0
    idle_sum: int = 0
    last_prune_timestamp: float = 0.0
    miner_running: bool = True
    director_running: bool = True

class OrchestratorService:
    def __init__(self):
        self.state = State()
        self._event_queue = asyncio.Queue(maxsize=100)
        self._job_queue: asyncio.Queue[tuple[JobType, Optional[GameEventModel]]] = asyncio.Queue(maxsize=100)
        self._stop = asyncio.Event()
        self._tasks = []
        # avoid ollama collapse: only 2 instances at the time
        self._llm_semaphore = asyncio.Semaphore(2)

    async def start(self):
        self._tasks = [
            asyncio.create_task(self._reducer_loop()),
            asyncio.create_task(self._scheduler_loop()),
            asyncio.create_task(self._worker_loop())
        ]

    async def stop(self):
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def publish(self, sig: Signal):
        await self._event_queue.put(sig)

    async def _get_last_event(self, trigger_timestamp: Optional[float]) -> Optional[GameEventModel]:
        if trigger_timestamp is None:
            return await state_buffer.get_last_event()
        return await state_buffer.get_last_event_before(trigger_timestamp)

    async def _reducer_loop(self):
        while not self._stop.is_set():
            sig = await self._event_queue.get()
            if sig.type == SignalType.EVENT_INGESTED:
                self.state.idle_sum += float(sig.value or 0)
            elif sig.type == SignalType.IDLE_INCREMENT:
                self.state.idle_sum += int(sig.value or 1)
            elif sig.type == SignalType.FORCE_DIRECTOR:
                last_event = await self._get_last_event(sig.timestamp)
                await self._enqueue_once(JobType.RUN_DIRECTOR, director_payload=last_event)

            if self.state.tension_sum >= 100 or self.state.idle_sum >= 10:
                last_event = await self._get_last_event(sig.timestamp)
                await self._enqueue_once(JobType.RUN_DIRECTOR, director_payload=last_event)
                self.state.tension_sum = 0
                self.state.idle_sum = 0

    async def _scheduler_loop(self):
        while not self._stop.is_set():
            await asyncio.sleep(5)
            now = time.time()
            if now - self.state.last_prune_timestamp > 5 * 24 * 60 * 60:
                await self._enqueue_once(JobType.RUN_PRUNER)
                self.state.last_prune_timestamp = now

    async def _enqueue_once(self, job_type: JobType, director_payload: Optional[GameEventModel] = None):
        # minimal coalescing strategy could track pending flags
        await self._job_queue.put((job_type, director_payload))

    async def _worker_loop(self):
        while not self._stop.is_set():
            job, director_payload = await self._job_queue.get()
            async with self._llm_semaphore:
                if job == JobType.RUN_MINER:
                    await run_miner()
                elif job == JobType.RUN_DIRECTOR:
                    if director_payload is not None:
                        await run_director(director_payload)
                elif job == JobType.RUN_PRUNER:
                    await run_pruner()