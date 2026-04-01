# the asyncio background loop
# watches the clock, flushes the buffer, calculates Tension Sum, evaluates importance score triggers
# dictates how llm and memory modules interact
import asyncio
from core import state_buffer

TENSION_SUM: int = 0

async def run_clock(interval: int):
    print(F"Clock started. Interval: {interval} seconds.")
