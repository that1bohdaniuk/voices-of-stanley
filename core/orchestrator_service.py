# the asyncio background loop
# watches the clock, flushes the buffer, calculates Tension Sum, evaluates importance score triggers
# dictates how llm and memory modules interact
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import config
from api.schemas import GameEventModel
from core import state_buffer
from llm.director import run_director
from llm.pruner import run_pruner
from llm.miner import run_miner

class SignalType(Enum):
    EVENT_INGESTED = auto()
    # IDLE_INCREMENT IS NOT IMPLEMENTED FOR NOW
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
    payload: Optional[GameEventModel] = None

@dataclass
class State:
    tension_sum: float = 0.0
    idle_sum: int = 0
    last_prune_timestamp: float = 0.0
    miner_running: bool = True
    director_running: bool = True

class OrchestratorService:
    def __init__(self):
        logging.info("[ORCHESTRATOR] Initializing Orchestrator Service.")
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
        logging.info("[ORCHESTRATOR] Stopping orchestrator service.")
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # use it to publish signals from the API or other parts of the system, e.g. when a new event is ingested
    async def publish(self, sig: Signal):
        await self._event_queue.put(sig)

    async def _enqueue_once(self, job_type: JobType, director_payload: Optional[GameEventModel] = None):
        # minimal coalescing strategy could track pending flags
        await self._job_queue.put((job_type, director_payload))

    async def _get_last_event(self, trigger_timestamp: Optional[float]) -> Optional[GameEventModel]:
        if trigger_timestamp is None:
            return await state_buffer.get_last_event()
        return await state_buffer.get_last_event_before(trigger_timestamp)

    async def _get_current_game_time(self) -> Optional[float]:
        last_event = await state_buffer.get_last_event()
        if last_event is None:
            return None
        return float(last_event.timestamp)

    async def _reducer_loop(self):
        while not self._stop.is_set():
            sig = await self._event_queue.get()
            if sig.type == SignalType.EVENT_INGESTED:
                logging.info("[ORCHESTRATOR] Signal EVENT_INGESTED.")
                self.state.idle_sum += float(sig.value or 0)
            elif sig.type == SignalType.IDLE_INCREMENT:
                logging.info("[ORCHESTRATOR] Signal IDLE_INCREMENT.")
                self.state.idle_sum += int(sig.value or 1)
            elif sig.type == SignalType.FORCE_DIRECTOR:
                logging.info("[ORCHESTRATOR] Signal FORCE_DIRECTOR.")
                if sig.payload:
                    trigger_event = sig.payload
                else:
                    trigger_event = await self._get_last_event(sig.timestamp)
                await self._enqueue_once(JobType.RUN_DIRECTOR, director_payload=trigger_event)

            if self.state.tension_sum >= config.TENSION_SUM_THRESHOLD or self.state.idle_sum >= config.IDLE_DIRECTOR_THRESHOLD:
                if sig.payload:
                    trigger_event = sig.payload
                else:
                    trigger_event = await self._get_last_event(sig.timestamp)
                await self._enqueue_once(JobType.RUN_DIRECTOR, director_payload=trigger_event)
                self.state.tension_sum = 0
                self.state.idle_sum = 0

    async def _scheduler_loop(self):
        while not self._stop.is_set():
            await asyncio.sleep(5)
            current_game_time = await self._get_current_game_time()
            if current_game_time is None:
                continue

            if current_game_time - self.state.last_prune_timestamp > config.EVENT_PRUNE_TIME_THRESHOLD_SECONDS:
                await self._enqueue_once(JobType.RUN_PRUNER)
                self.state.last_prune_timestamp = current_game_time

    async def _worker_loop(self):
        while not self._stop.is_set():
            job, director_payload = await self._job_queue.get()
            async with self._llm_semaphore:
                if job == JobType.RUN_MINER:
                    logging.info("[ORCHESTRATOR] Job RUN_MINER started.")
                    await run_miner()
                elif job == JobType.RUN_DIRECTOR:
                    if director_payload is not None:
                        logging.info("[ORCHESTRATOR] Job RUN_DIRECTOR started.")
                        await run_director(director_payload)
                elif job == JobType.RUN_PRUNER:
                    logging.info("[ORCHESTRATOR] Job RUN_PRUNER started.")
                    await run_pruner()