# test_routes_and_buffer.py
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from main import app
from core import state_buffer
from api.schemas import GameEventModel

client = TestClient(app)

@pytest_asyncio.fixture(autouse=True)
async def clear_buffer():
    """ensure the state buffer is empty before each test."""
    await state_buffer.flush()


@pytest.mark.asyncio
async def test_state_buffer_append_and_flush():
    """test that the buffer safely appends and flushes events."""
    event1 = GameEventModel(label="Door opened", timestamp=100.0)
    event2 = GameEventModel(label="Footsteps", timestamp=105.0)

    await state_buffer.append(event1)
    await state_buffer.append(event2)

    flushed_events = await state_buffer.flush()
    assert len(flushed_events) == 2
    assert flushed_events[0].label == "Door opened"

    empty_flush = await state_buffer.flush()
    assert len(empty_flush) == 0


def test_http_root():
    """test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Stanley API is running."}


def test_http_debug_state():
    """test the debug state endpoint."""
    response = client.get("/debug/state")
    assert response.status_code == 200
    assert response.json()["status"] == "Listening"


@pytest.mark.asyncio
async def test_http_ping():
    """test the ping endpoint and verify it hits the state buffer."""
    payload = {
        "label": "Radar ping detected",
        "timestamp": 12345.6,
        "location": "Server Room",
        "importance": 5.0
    }

    response = client.post("/ping", json=payload)
    assert response.status_code == 200

    flushed = await state_buffer.flush()
    assert len(flushed) == 1
    assert flushed[0].label == "Radar ping detected"


import pytest
import json
from fastapi.testclient import TestClient

from main import app
from api import routes_ws

client = TestClient(app)


def test_websocket_connection_and_receive():
    """test that the websocket accepts connections and receives data."""

    with client.websocket_connect("/ws") as websocket:
        assert routes_ws.active_connection is not None

        test_payload = {"event": "player_moved", "x": 10, "y": 20}
        websocket.send_text(json.dumps(test_payload))

    # once we exit the 'with' block, the test client automatically disconnects
    assert routes_ws.active_connection is None


@pytest.mark.asyncio
async def test_send_action_to_game():
    """test that the server can push actions back down the websocket."""

    with client.websocket_connect("/ws") as websocket:
        action_payload = {"action": "spawn_entity", "entity_id": "alien_1"}

        await routes_ws.send_action_to_game(action_payload)

        received_data = websocket.receive_json()
        assert received_data == action_payload


@pytest.mark.asyncio
async def test_send_action_no_connection():
    """test that sending handles if the game isn't connected."""

    routes_ws.active_connection = None

    action_payload = {"action": "explode_server"}
    await routes_ws.send_action_to_game(action_payload)