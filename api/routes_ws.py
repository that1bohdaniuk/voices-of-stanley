# defines the fastapi websocket endpoint
# catches incoming game deltas, passes them to the buffer, and maintains the active connection object to send payloads back to the C++ mod.