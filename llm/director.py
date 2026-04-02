# contains system prompt for director qwen3.5 model
# ingests memories and psychoprofile to generate a final JSON payload
import json

from ollama import AsyncClient

import config
from llm.client import check_ollama_server, unload_ollama_model
from api.schemas import GameEventModel, DirectorEventModel
from api.routes_ws import send_action_to_game
from memory.archive import retrieve

client = AsyncClient()

async def _prepare_context(_trigger_event: GameEventModel) -> str:
    # retrieve information from archive and psychoprofile, format it into a prompt context for the director model
    # (retrieve returns similar events to passed one)
    prompt_string: str
    _similar_events = await retrieve(_trigger_event)
    with open("../data/psychoprofile.json") as f:
        _psychoprofile_json = json.load(f)
    _psychoprofile_string = json.dumps(_psychoprofile_json, indent=2)

    prompt_string = f'''Trigger event: {_trigger_event}\n
    Similar events: {', '.join(str(_event) for _event in _similar_events)}\n
    Psychoprofile: {_psychoprofile_string}\n'''

    return prompt_string


async def director(_trigger_event: GameEventModel):
    # call the director model, feed it context from _prepare_context, and return the final JSON output
    await check_ollama_server()
    _model = config.DIRECTOR_MODEL

    try:
        prompt_string = await _prepare_context(_trigger_event)

        #setup ollama call
        response = await client.generate(
            model=_model,
            prompt=prompt_string,
            format=DirectorEventModel.model_json_schema()
        )
        response = DirectorEventModel.model_validate_json(response['response'])
        response_json = response.model_dump(mode='json')

        await send_action_to_game(response_json)

    finally:
        await unload_ollama_model(_model)
