# the asyncio background loop
# watches the clock, flushes the buffer, calculates Tension Sum, evaluates importance score triggers
# dictates how llm and memory modules interact
import asyncio
import logging

import config
from llm.miner import mine_buffer
from core import state_buffer

TENSION_SUM: int = 0
IDLE_SUM: int = 0

async def update_tension_sum(amount: float):
    global TENSION_SUM
    TENSION_SUM += amount

async def update_idle_sum(amount: int):
    global IDLE_SUM
    IDLE_SUM += amount


async def loop(interval: int):
    global TENSION_SUM, IDLE_SUM
    logging.info(F"Loop started. Interval: {interval} seconds.")
    while True:
        await asyncio.sleep(interval)
        _last_event = state_buffer.get_last_event()
        # look if state_buffer threshold is reached; tension is accumulated enough; idle threshold is reached;
        # if so flush it and send to miner
        if TENSION_SUM >= config.TENSION_SUM_THRESHOLD or IDLE_SUM >= config.IDLE_DIRECTOR_THRESHOLD :
            await mine_buffer()
            TENSION_SUM = 0
            IDLE_SUM = 0




