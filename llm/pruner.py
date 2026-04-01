# script that assembles psychoprofile of a person from events that occurred.
import json
import logging

from ollama import AsyncClient
from api.schemas import ProfileJSONModel
from llm.client import check_ollama_server
from memory.archive import get_all_to_prune_events, delete_events_by_id

client = AsyncClient()

async def prune():
    await check_ollama_server()

    try:
        _events = await get_all_to_prune_events()
        _events_string = json.dumps(_events)

        _ids_to_delete = _events["ids"]

    except Exception as e:
        logging.error(f"Failed to fetch events: {e}")
        return


    try:
        with open("../data/psychoprofile.json") as f:
            _data = json.load(f)
        _data_string = json.dumps(_data)
    except FileNotFoundError:
        logging.warning("No existing psychoprofile found. Starting fresh.")
        _data = {"deltas": {}}
        _data_string = "{}"
    except json.JSONDecodeError:
        logging.error("psychoprofile.json is corrupted.")
        return

    _response = await client.generate(
        #model="pruner-9B",
        model="qwen3.5:0.8B",
        prompt=("Player's psychoprofile: " + _data_string + "\nEvents to assess:\n" + _events_string),
        format=ProfileJSONModel.model_json_schema()
    )
    _raw_json = _response['response']


    try:
        _profile_data = ProfileJSONModel.model_validate(_data)
        _generated_data = ProfileJSONModel.model_validate_json(_raw_json)

        for key, value in _generated_data.deltas.items():
            current_val = _profile_data.deltas.get(key, 0)
            _profile_data.deltas[key] = current_val + value
    except Exception as e:
        logging.error(f"Failed to validate psychoprofile data: {e}")
        return

    try:
        with open("data/psychoprofile.json", "w", encoding='utf-8') as f:
            json.dump(_profile_data.model_dump(), f, indent=4, ensure_ascii=False)
            await delete_events_by_id(ids=_ids_to_delete)

    except Exception as e:
        logging.error(f"Failed to save psychoprofile data: {e}")

