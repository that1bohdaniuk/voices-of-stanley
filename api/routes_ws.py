# defines the fastapi websocket endpoint
# catches incoming game deltas, passes them to the buffer, and maintains the active connection object to send payloads back to the C++ mod.
import json
from rich.json import JSON
from fastapi import APIRouter, WebSocketDisconnect, WebSocket

from api.schemas import GameEventModel
from memory.archive import embed

ws_router = APIRouter()

#global reference so other modules (like director) can push data back into game
active_connection: WebSocket

@ws_router.websocket("/ws")
async def game_endpoint(websocket: WebSocket):
    print("Websocket triggered.")
    # db = websocket.app.state.vector_db
    global active_connection
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            # printing raw_data for logging because it's more viewable
            print(f"Websocket triggered. Received: {raw_data}")

    except WebSocketDisconnect:
        print("VotV disconnected.")
        active_connection = None
        await websocket.close()


async def send_action_to_game(action_payload: JSON):
    # call in director
    if active_connection:
        await active_connection.send_json(action_payload)
    else:
        print("Tried to send an action to VotV, but connection closed.")