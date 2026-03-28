# contains system prompts and few-shot examples for miner qwen3.5:4B model
# formats the raw buffer data into discrete event chunks and assigns the 1-10 importance score
import asyncio
import json

from api.schemas import GameEventModel, EventExtractionModel
from core import state_buffer
from ollama import AsyncClient

client = AsyncClient()

async def mine_buffer() -> EventExtractionModel:
  # receives flushed buffer of raw data and outputs structured GameEvent chunks,
  #  combining semantically similar events and neglecting noise
  _events: list[GameEventModel] = await state_buffer.flush()
  _data_string = json.dumps(_events, indent=2)

  response  = await client.chat(
    model="qwen3.5:0.8B",
    messages=[{'role': 'user', 'content': ('Raw data: '+ _data_string)}],
    format=EventExtractionModel.model_json_schema()
  )

  return EventExtractionModel.model_validate_json(response.message.content)
