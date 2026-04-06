# test_llm_modules.py
# these tests are not using ollama service, they just test the hardcoded stuff happening
import json
from unittest.mock import patch, AsyncMock, mock_open

import pytest

from api.schemas import GameEventModel
from core import state_buffer
from llm import miner, pruner


@pytest.mark.asyncio
@patch("llm.miner.client.chat", new_callable=AsyncMock)
@patch("llm.miner.embed_bunch", new_callable=AsyncMock)
async def test_mine_buffer(mock_embed_bunch, mock_chat):
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
    result = await miner.run_miner()

    assert len(result.extracted_events) == 1
    assert result.extracted_events[0].label == "Processed movement chunk"
    mock_embed_bunch.assert_awaited_once()

    remaining_buffer = await state_buffer.flush()
    assert len(remaining_buffer) == 0


@pytest.mark.asyncio
@patch("llm.pruner.delete_events_by_id", new_callable=AsyncMock)
@patch("llm.pruner.get_all_to_prune_events", new_callable=AsyncMock)
@patch("llm.pruner.client.generate", new_callable=AsyncMock)
@patch("llm.pruner.check_ollama_server", new_callable=AsyncMock)
@patch("builtins.open", new_callable=mock_open, read_data='{"deltas": {"paranoia": 10}}')
async def test_pruner(mock_file, mock_check_ollama, mock_generate, mock_get_events, mock_delete):
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
    await pruner.run_pruner()

    mock_file().write.assert_called()

    # intercept the data written to the file to ensure math was done correctly
    written_data = "".join(call.args[0] for call in mock_file().write.mock_calls)
    saved_json = json.loads(written_data)

    # base was 10, LLM added 5 -> Should be 15
    assert saved_json["deltas"]["paranoia"] == 15
    # new trait added
    assert saved_json["deltas"]["aggression"] == 2

    # verify that the prune command instructed ChromaDB to delete the processed IDs
    mock_check_ollama.assert_called_once()
    mock_delete.assert_called_once_with(ids=["uuid-1", "uuid-2"])


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data='{"deltas": {"anger": 15, "suspicion": 8}}')
@patch("llm.director.check_ollama_server", new_callable=AsyncMock)
@patch("llm.director.retrieve", new_callable=AsyncMock)
@patch("llm.director.client.generate", new_callable=AsyncMock)
@patch("llm.director.send_action_to_game", new_callable=AsyncMock)
@patch("llm.director.unload_ollama_model", new_callable=AsyncMock)
async def test_director(mock_unload, mock_send_action, mock_generate, mock_retrieve, mock_check_ollama, mock_file):
    """test that director retrieves context, calls LLM, validates response, and sends to game."""
    from llm import director

    # Mock config to use a test model name
    with patch("llm.director.config") as mock_config:
        mock_config.DIRECTOR_MODEL = "director-9B"

        # Setup trigger event
        trigger_event = GameEventModel(
            label="Player entered mysterious room",
            timestamp=1000.0,
            location="Observatory",
            importance=8.0
        )

        mock_retrieve.return_value = [
            {'id': 'evt-1', 'document': 'Strange signal detected', 'score': 0.9},
            {'id': 'evt-2', 'document': 'Power outage in lab', 'score': 0.7}
        ]

        import uuid
        mock_llm_response = {
            "id": str(uuid.uuid4()),
            "type": "spawn_entity",
            "data": {"entity_type": "ghost", "location": "Observatory", "behavior": "haunting"}
        }
        mock_generate.return_value = {'response': json.dumps(mock_llm_response)}

        await director.run_director(trigger_event)

        mock_check_ollama.assert_called_once()
        mock_retrieve.assert_called_once_with(trigger_event)

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs['model'] == 'director-9B'
        assert 'Trigger event:' in call_kwargs['prompt']
        assert 'Similar events:' in call_kwargs['prompt']
        assert 'Psychoprofile:' in call_kwargs['prompt']

        mock_send_action.assert_called_once()
        sent_payload = mock_send_action.call_args.args[0]
        assert isinstance(sent_payload, dict), "Payload should be dict, not JSON string"
        assert sent_payload['type'] == 'spawn_entity'
        assert sent_payload['data']['entity_type'] == 'ghost'

        mock_unload.assert_called_once_with('director-9B')


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open, read_data='{}')
@patch("llm.director.check_ollama_server", new_callable=AsyncMock)
@patch("llm.director.retrieve", new_callable=AsyncMock)
@patch("llm.director.client.generate", new_callable=AsyncMock)
@patch("llm.director.send_action_to_game", new_callable=AsyncMock)
@patch("llm.director.unload_ollama_model", new_callable=AsyncMock)
async def test_director_handles_validation_error(mock_unload, mock_send_action, mock_generate, mock_retrieve, mock_check_ollama, mock_file):
    """test that director still unloads model even if LLM validation fails."""
    from llm import director
    from pydantic import ValidationError

    with patch("llm.director.config") as mock_config:
        mock_config.DIRECTOR_MODEL = "director-9B"

        trigger_event = GameEventModel(label="Test", timestamp=100.0)
        mock_retrieve.return_value = []

        invalid_response = {"id": "123"}  # missing 'type' and 'data'
        mock_generate.return_value = {'response': json.dumps(invalid_response)}

        with pytest.raises(ValidationError):
            await director.run_director(trigger_event)

        mock_unload.assert_called_once_with('director-9B'
                                            )
        # send_action_to_game should NOT be called on validation error
        mock_send_action.assert_not_called()

