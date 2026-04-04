# wraps the chromadb client
# handles the embedding of miner's chunks and executes time-decay and importance cosine similarity formula during retrieval
import asyncio
import chromadb
import numpy as np
import time
from typing import Any, List, cast
from chromadb import QueryResult, GetResult
from chromadb.api.types import Metadata, OneOrMany

import config
from api.schemas import GameEventModel

_collection: chromadb.Collection


def _flatten_metadata_rows(metadatas: Any) -> list[dict]:
    if not metadatas:
        return []
    if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
        return [row for group in metadatas for row in (group or []) if isinstance(row, dict)]
    if isinstance(metadatas, list):
        return [row for row in metadatas if isinstance(row, dict)]
    return []


def _resolve_event_time(metadatas: Any, fallback_to_wallclock: bool = True) -> float | None:
    rows = _flatten_metadata_rows(metadatas)
    timestamps: list[float] = []
    for row in rows:
        value = row.get("timestamp")
        if isinstance(value, (int, float)):
            timestamps.append(float(value))

    if timestamps:
        return max(timestamps)
    return time.time() if fallback_to_wallclock else None

async def initialize_chroma_client() -> chromadb.ClientAPI:
    global _collection

    _client = chromadb.PersistentClient(path="./data/")
    _collection = _client.get_or_create_collection(
        name="events-collection",
        # explicitly define hnsw vector space as cosine
        metadata={"hnsw:space": "cosine"})

    return _client


async def embed(_event: GameEventModel):
    print(f"Embedding '{_event.label}'...")
    _metadatas: dict[str, object] = {
        'timestamp': _event.timestamp,
        'importance': _event.importance,
        **{str(k): v for k, v in (_event.details or {}).items()}
    }

    if _event.location:
        _metadatas['location'] = _event.location

    await asyncio.to_thread(_collection.add,
        ids=[str(_event.id)],
        documents=[_event.label],
        metadatas=cast(OneOrMany[Metadata], cast(object, [_metadatas])))


async def embed_bunch(_events: List[GameEventModel]):
    global _collection

    if not _events:
        return

    _ids = [str(e.id) for e in _events]
    _documents = [e.label for e in _events]

    _metadatas = []
    for e in _events:
        meta: dict[str, object] = {
            'timestamp': e.timestamp,
            'importance': e.importance,
            **{str(k): v for k, v in (e.details or {}).items()}
        }
        if e.location:
            meta['location'] = e.location
        _metadatas.append(meta)

    await asyncio.to_thread(
        _collection.add,
        ids=_ids,
        documents=_documents,
        metadatas=cast(list[Metadata], cast(object, _metadatas))
    )


#Time-Weighted RAG with importance score (loop up miro)
async def twrag(_chroma_results: QueryResult, broadness: int, current_time: float):
    if not _chroma_results["ids"] or not _chroma_results["ids"][0]:
        return []
    _decay_rate = config.TWRAG_DECAY_RATE
    _twrag_results = []

    _ids = _chroma_results["ids"][0]
    _metadatas = _chroma_results["metadatas"][0]
    _documents = _chroma_results["documents"][0]
    _distances = _chroma_results["distances"][0]

    for i in range(len(_ids)):
        _cos_similarity = 1 - _distances[i]
        _doc_timestamp = _metadatas[i].get('timestamp', 0)
        _age_seconds = max(0.0, float(current_time) - float(_doc_timestamp))
        _importance = max(_metadatas[i].get('importance', 1), 0.0001)

        _multiplier = np.exp(-(_decay_rate/_importance) * _age_seconds)
        _score = _cos_similarity * _multiplier
        _twrag_results.append({
            'id': _ids[i],
            'metadata': _metadatas[i],
            'document': _documents[i],
            'original_distance': _distances[i],
            'score': _score,
        })
    _sorted_twrag = sorted(_twrag_results, key=lambda x: x['score'], reverse=True)
    return _sorted_twrag[:broadness]



async def retrieve(_event: GameEventModel):
    # custom formula (look up miro whiteboard)
    global _collection
    # casting to custom Include type (Literal["embeddings", "documents"...]) so linter won't tell anything
    include_params = cast(chromadb.Include, ["embeddings", "metadatas", "documents", "distances"])
    # passing to thread because PersistentClient and collection.query() are synchronous and CPU-bound
    # query is did in c++ thread, so GIL is free and python can run in parallel
    results = await asyncio.to_thread(
        _collection.query,
        query_texts=[_event.label],
        n_results=50,
        include=include_params)
    raw_broadness = getattr(config, "TWRAG_RETRIEVAL_BROADNESS", 5)
    broadness = raw_broadness if isinstance(raw_broadness, int) and raw_broadness > 0 else 5

    return await twrag(
        _chroma_results=results,
        broadness=broadness,
        current_time=float(_event.timestamp),
    )
# usage example: retrieve(event, filter_dict={"event_type": "signal_detected"})


async def purge_events(current_timestamp: float | None = None):
    global _collection

    where_params = cast(chromadb.Where, {"importance": {"$lte": config.EVENT_PURGE_IMPORTANCE_THRESHOLD}})
    include_params = cast(chromadb.Include, ["metadatas"])

    result = await asyncio.to_thread(
        _collection.get,
        include=include_params,
        where = where_params
    )

    _ids = result["ids"]
    _metadatas = result["metadatas"]

    if not _ids:
        return

    _ids_to_delete = []
    _current_timestamp = current_timestamp if current_timestamp is not None else _resolve_event_time(_metadatas)
    if _current_timestamp is None:
        return


    for i in range(len(_ids)):
        _doc_timestamp = float(_metadatas[i].get('timestamp', 0))
        _delta_seconds = _current_timestamp - _doc_timestamp
        _delta_minutes = _delta_seconds / 60

        if _delta_minutes > config.EVENT_PURGE_TIME:
            _ids_to_delete.append(_ids[i])

    if not _ids_to_delete:
        return

    await asyncio.to_thread(
        _collection.delete,
        ids=_ids_to_delete,
    )

# function that returns all recent non-pruned events
async def get_all_to_prune_events():
    global _collection
    all_events = await asyncio.to_thread(
        _collection.get,
        include=cast(chromadb.Include, ["metadatas"])
    )
    current_timestamp = _resolve_event_time(all_events.get("metadatas"), fallback_to_wallclock=False)
    if current_timestamp is None:
        return {"ids": [], "metadatas": [], "documents": []}

    cutoff_timestamp = current_timestamp - config.EVENT_PRUNE_TIME_THRESHOLD_SECONDS

    where_params = cast(
        chromadb.api.types.Where,
        {"timestamp": {"$lt": cutoff_timestamp}}
    )
    include_params = cast(
        chromadb.api.types.Include,
        ["metadatas", "documents"]
    )

    result: GetResult = await asyncio.to_thread(
        _collection.get,
        where=where_params,
        include=include_params
    )

    return result

async def delete_events_by_id(ids: OneOrMany[chromadb.IDs]):
    global _collection

    await asyncio.to_thread(
        _collection.delete,
        ids=ids,
    )


