# sets up async Ollama client
import subprocess
import httpx
import ollama
import asyncio
import os

_ollama_process: subprocess.Popen | None = None
client = ollama.AsyncClient()

async def start_ollama_server(retries=30, delay=1):
    global _ollama_process
    try:
        # try to start a subprocess
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )
        for attempt in range(retries):
            # healthcheck loop to see if server is up, with retries and delay
            try:
                async with httpx.AsyncClient() as healthcheck:
                    await healthcheck.get("http://localhost:11434/api/version", timeout=2)
                    print("[Ollama] Server started succesfully.")
                    return True
            except httpx.ConnectError:
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
        # if we exhaust retries, terminate the process and raise error
        _ollama_process.terminate()
        raise RuntimeError("Ollama server failed to start.")
    except FileNotFoundError:
        raise RuntimeError("Ollama not found in PATH. Install from https://ollama.ai.")


async def stop_ollama_server():
    global _ollama_process
    # if process object exists, then terminate()
    # force kill if terminate fails
    if _ollama_process:
        _ollama_process.terminate()
        try:
            _ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ollama_process.kill()


async def check_ollama_server():
    global _ollama_process
    # process object existence check
    if _ollama_process:
        if _ollama_process.poll() is not None:
            print("[Ollama] Process exists but is dead. Restarting...")
            _ollama_process = None
        else:
            # healtcheck
            try:
                async with httpx.AsyncClient() as healthcheck:
                    await healthcheck.get("http://localhost:11434/api/version", timeout=2)
                    return True
            except httpx.ConnectError:
                print("[Ollama] Process running but server not responding.")
                _ollama_process.terminate()
                _ollama_process = None
    # if we got here then restarting server
    return await start_ollama_server()
