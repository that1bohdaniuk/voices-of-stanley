# sets up async Ollama client
import logging
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
            except httpx.HTTPError:
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
            except httpx.HTTPError:
                print("[Ollama] Process running but server not responding.")
                _ollama_process.terminate()
                _ollama_process = None
    # if we got here then restarting server
    return await start_ollama_server()


# TODO: write an auto-unload wrapper that accepts model and llm_call_coro as parameters
async def unload_ollama_model(model_name: str):
    try:
        async with httpx.AsyncClient() as hc:
            response = await hc.post(
                # see ollama api generate endpoint
                # https://docs.ollama.com/api/generate#body-keep-alive-one-of-0
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=5
            )
            logging.info(f"[Ollama] Model unloaded succesfully.")
    except Exception as e:
        logging.warning(f"[Ollama] Failed to unload ollama model: {e}")