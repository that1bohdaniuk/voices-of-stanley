# contains system prompts and few-shot examples for miner qwen3.5:4B model
# formats the raw buffer data into discrete event chunks and assigns the 1-10 importance score
import json

from api.schemas import GameEventModel, EventExtractionModel
from config import FORCE_DIRECTOR_THRESHOLD
from core import state_buffer
from ollama import AsyncClient

import config
from llm.client import check_ollama_server, unload_ollama_model
from memory.archive import embed_bunch

client = AsyncClient()

async def run_miner():
  # receives flushed buffer of raw data and outputs structured GameEvent chunks,
  #  combining semantically similar events and neglecting noise
  await check_ollama_server()
  _model = config.MINER_MODEL

  try:
    _events: list[GameEventModel] = await state_buffer.flush()
    if not _events:
      raise ValueError("No events flushed from buffer when miner called.")
    _data_string = json.dumps([event.model_dump(mode='json') for event in _events], indent=2)

    response  = await client.chat(
      model=_model,
      messages=[{'role': 'user', 'content': ('Raw data: '+ _data_string)}],
      format=EventExtractionModel.model_json_schema()
    )

    events_to_embed = EventExtractionModel.model_validate_json(response.message.content)

    for event in events_to_embed.extracted_events:
      if event.importance >= FORCE_DIRECTOR_THRESHOLD:
        from core.orchestrator_service import Signal, SignalType
        from main import app

        await app.state.orchestrator.publish(Signal(type=SignalType.FORCE_DIRECTOR, payload=event, value=0, timestamp=event.timestamp))

    await embed_bunch(events_to_embed.extracted_events)
    return events_to_embed

  finally:
    await unload_ollama_model(_model)