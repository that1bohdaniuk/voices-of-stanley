# defines the fastapi websocket endpoint
# catches incoming game deltas, passes them to the buffer, and maintains the active connection object to send payloads back to the C++ mod.
import json
import logging

from fastapi import APIRouter, WebSocketDisconnect, WebSocket
from pydantic import ValidationError

from api.schemas import GameEventModel
from core import state_buffer

ws_router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#global reference so other modules (like director) can push data back into game
active_connection: WebSocket | None = None

@ws_router.websocket("/ws")
async def game_endpoint(websocket: WebSocket):
    print("[WS] Connection attempt.")
    global active_connection
    active_connection = websocket

    await websocket.accept()
    print("[WS] Connection accepted.")
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data_list = json.loads(raw_data)
                data = [item for item in data_list]
            except json.JSONDecodeError as e:
                logger.error(f"[WS] Invalid JSON: {raw_data}")
                continue

            # validate incoming json against GameEventModel schema
            try:
                for chunk in data:
                    event = GameEventModel(**chunk)
                    await state_buffer.append(event)
                    # make sure that orchestrator knows that we've received an event
                    from core.orchestrator_service import Signal, SignalType

                    await websocket.app.state.orchestrator.publish(Signal(type=SignalType.EVENT_INGESTED, value=1))
                    logger.debug(f"[WS] Event received: {event.label} (importance={event.importance})")

            except ValidationError as ve:
                logger.error(f"[WS] Invalid event payload: {ve}")
                continue
            except Exception as e:
                logger.error(f"[WS] Unexpected error processing this event: {e}")
                continue

    except WebSocketDisconnect:
        print("VotV disconnected.")
        active_connection = None
        await websocket.close()
    except Exception as e:
        logger.error(f"[WS] Unexpected error: {e}")
        active_connection = None
        await websocket.close()


async def send_action_to_game(action_payload):
    # function to be called in director
    if active_connection:
        try:
            await active_connection.send_json(action_payload)
            logger.debug(f"[WS] Action sent to game endpoint: {action_payload}")
        except Exception as e:
            logger.error(f"[WS] Failed to send action: {e}")
    else:
        logger.warning("Tried to send an action to VotV, but connection closed.")