# test_llm_modules.py
# these tests are not using ollama service, they just test the hardcoded stuff happening
import pytest
import json
from unittest.mock import patch, AsyncMock, mock_open

from llm import miner, pruner
from api.schemas import GameEventModel
from core import state_buffer


@pytest.mark.asyncio
@patch("llm.miner.client.chat", new_callable=AsyncMock)
async def test_mine_buffer(mock_chat):
    """test that the miner correctly pulls from the buffer and parses LLM output."""

    raw_event = GameEventModel(label="Raw movement data", timestamp=100.0)
    await state_buffer.append(raw_event)

    # mock the Ollama response to return a valid JSON string matching EventExtractionModel
    mock_llm_response = {
        "extracted_events": [
            {
                "label": "Processed movement chunk",
                "timestamp": 100.0,
                "importance": 4.5
            }
        ]
    }

    class MockMessage:
        content = json.dumps(mock_llm_response)

    class MockResponse:
        message = MockMessage()

    mock_chat.return_value = MockResponse()

    # run the miner
    result = await miner.mine_buffer()

    assert len(result.extracted_events) == 1
    assert result.extracted_events[0].label == "Processed movement chunk"

    remaining_buffer = await state_buffer.flush()
    assert len(remaining_buffer) == 0


@pytest.mark.asyncio
@patch("llm.pruner.delete_events_by_id", new_callable=AsyncMock)
@patch("llm.pruner.get_all_to_prune_events", new_callable=AsyncMock)
@patch("llm.pruner.client.generate", new_callable=AsyncMock)
@patch("builtins.open", new_callable=mock_open, read_data='{"deltas": {"paranoia": 10}}')
async def test_pruner(mock_file, mock_generate, mock_get_events, mock_delete):
    """test the pruner reads the file, calls the LLM, updates the profile, and deletes old events."""

    # mock the ChromaDB retrieval
    mock_get_events.return_value = {
        "ids": [["uuid-1", "uuid-2"]],
        "documents": [["Event 1", "Event 2"]],
        "metadatas": [[{"timestamp": 100}, {"timestamp": 200}]]
    }

    # mock the LLM output
    mock_llm_response = {
        "deltas": {
            "paranoia": 5,
            "aggression": 2
        }
    }
    mock_generate.return_value = {'response': json.dumps(mock_llm_response)}

    # pruner
    await pruner.prune()

    mock_file().write.assert_called()

    # intercept the data written to the file to ensure math was done correctly
    written_data = "".join(call.args[0] for call in mock_file().write.mock_calls)
    saved_json = json.loads(written_data)

    # base was 10, LLM added 5 -> Should be 15
    assert saved_json["deltas"]["paranoia"] == 15
    # new trait added
    assert saved_json["deltas"]["aggression"] == 2

    # verify that the prune command instructed ChromaDB to delete the processed IDs
    mock_delete.assert_called_once_with(ids=[["uuid-1", "uuid-2"]])