# wraps the chromadb client
# handles the embedding of miner's chunks and executes time-decay and importance cosine similarity formula during retrieval
import asyncio
import time
import uuid
import chromadb
import numpy as np
from typing import List, cast
from chromadb import QueryResult
import config
from api.schemas import GameEventModel


_collection: chromadb.Collection

async def initialize_chroma_client() -> chromadb.ClientAPI:
    global _collection

    _client = chromadb.PersistentClient(path="./data/")
    _collection = _client.get_or_create_collection(
        name="events-collection",
        # explicitly define hsnw vector space as cosine
        metadata={"hsnw:space": "cosine"})

    return _client


async def embed(_event: GameEventModel):
    print(f"Embedding '{_event.label}'...")
    _metadatas = {
        'timestamp': _event.timestamp.timestamp(),
        'importance': _event.importance,
        **{str(k): v for k, v in _event.details.items()}
    }
    await asyncio.to_thread(_collection.add,
        ids=[str(uuid.uuid4())],
        documents=[_event.label],
        metadatas=[_metadatas])


async def embed_bunch(_events: List[GameEventModel]):
    global _collection

    _ids = [str(uuid.uuid4()) for _ in _events]
    _documents = [e.label for e in _events]
    _metadatas = [{'timestamp': e.timestamp.timestamp(),
                   'importance': e.importance,
                   **{str(k): v for k, v in e.details.items()}
                   } for e in _events]

    await asyncio.to_thread(
        _collection.add,
        ids=_ids,
        documents=_documents,
        metadatas=_metadatas
    )


#Time-Weighted RAG with importance score (loop up miro)
async def twrag(_chroma_results: QueryResult, broadness: int):
    if not _chroma_results["ids"] or not _chroma_results["ids"][0]:
        return []
    _decay_rate = config.TWRAG_DECAY_RATE
    _twrag_results = []
    _current_time = time.time()

    _ids = _chroma_results["ids"][0]
    _metadatas = _chroma_results["metadatas"][0]
    _documents = _chroma_results["documents"][0]
    _distances = _chroma_results["distances"][0]

    for i in range(len(_ids)):
        _cos_similarity = 1 - _distances[i]
        _doc_timestamp = _metadatas[i].get('timestamp', 0)
        _age_seconds = _current_time - _doc_timestamp
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



async def retrieve(_event: GameEventModel, filter_dict: dict = None):
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
        include=include_params,
        where=filter_dict)
    return await twrag(_chroma_results=results, broadness=5)

# usage example: retrieve(event, filter_dict={"event_type": "signal_detected"})

