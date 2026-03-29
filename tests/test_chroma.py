import pytest
import chromadb
from unittest.mock import patch
import uuid
from memory import archive
from api.schemas import GameEventModel


# --- FIXTURES ---

@pytest.fixture(autouse=True)
def setup_ephemeral_db():
    #overrides the PersistentClient with an in-memory client
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="test-events-collection",
        metadata={"hsnw:space": "cosine"}
    )
    # inject the ephemeral collection
    archive._collection = collection
    yield collection
    # teardown
    client.delete_collection("test-events-collection")


@pytest.fixture
def mock_config():
    # mocks the config variables used
    with patch("memory.archive.config") as mock_conf:
        mock_conf.TWRAG_DECAY_RATE = 0.01
        mock_conf.CURRENT_TIME = 1000.0
        mock_conf.EVENT_PURGE_IMPORTANCE_THRESHOLD = 5.0
        mock_conf.EVENT_PURGE_TIME = 60.0  # minutes
        yield mock_conf


# --- TESTS ---

@pytest.mark.asyncio
async def test_embed_empty_bunch():
    """Ensure the system gracefully handles an empty list of events."""
    await archive.embed_bunch([])
    results = archive._collection.get()
    assert len(results["ids"]) == 0

@pytest.mark.asyncio
async def test_retrieve_empty_collection(mock_config):
    """Retrieving from an empty collection should return an empty list, not crash."""
    query_event = GameEventModel(
        label="Is anyone there?",
        timestamp=mock_config.CURRENT_TIME,
        importance=1.0
    )
    results = await archive.retrieve(query_event)
    # Assuming your implementation returns an empty list when no docs exist
    assert results == []

@pytest.mark.asyncio
async def test_embed_single_event():
    event = GameEventModel(
        id=uuid.uuid4(),
        label="Player heard a strange sound.",
        timestamp=900.0,
        location="Radar Tower",
        importance=3.0,
        details={"volume": "loud"}
    )

    await archive.embed(event)

    # verify it was inserted
    results = archive._collection.get()
    assert len(results["ids"]) == 1
    assert results["ids"][0] == str(event.id)
    assert results["documents"][0] == event.label
    assert results["metadatas"][0]["location"] == "Radar Tower"


@pytest.mark.asyncio
async def test_embed_bunch():
    events = [
        GameEventModel(label="Signal 1", timestamp=100.0, importance=1.0),
        GameEventModel(label="Signal 2", timestamp=200.0, importance=2.0)
    ]

    await archive.embed_bunch(events)

    results = archive._collection.get()
    assert len(results["ids"]) == 2


@pytest.mark.asyncio
async def test_twrag_retrieval_logic(mock_config):
    # We will exaggerate the values to ensure the TWRAG math forcefully overrides
    # any slight base cosine similarity differences from the embedding model.

    # Event 1: slightly old (age 200), but MAX importance (10.0)
    # exp(-(0.01 / 10.0) * 200) = ~0.81 multiplier
    event1 = GameEventModel(label="Alien encounter at tower", timestamp=800.0, importance=10.0)

    # Event 2: very recent (age 50), but MIN importance (0.1)
    # exp(-(0.01 / 0.1) * 50) = ~0.006 multiplier
    event2 = GameEventModel(label="Wind blowing at tower", timestamp=950.0, importance=0.1)

    await archive.embed_bunch([event1, event2])

    # Query for "tower"
    query_event = GameEventModel(label="tower", timestamp=mock_config.CURRENT_TIME, importance=1.0)
    results = await archive.retrieve(query_event)

    assert len(results) == 2
    # the multiplier for the alien encounter (~0.81) will completely crush
    # the multiplier for the wind blowing (~0.006).
    assert results[0]['document'] == "Alien encounter at tower"


@pytest.mark.asyncio
async def test_twrag_pure_time_decay(mock_config):
    """
    test that when two events have the exact same importance and similarity,
    the more recent event wins due to the time decay penalty.
    """
    # both are identical in content and importance, but differ in age.
    event_old = GameEventModel(label="Footsteps", timestamp=500.0, importance=5.0)
    event_new = GameEventModel(label="Footsteps", timestamp=950.0, importance=5.0)

    await archive.embed_bunch([event_old, event_new])

    query_event = GameEventModel(
        label="Footsteps",
        timestamp=mock_config.CURRENT_TIME,  # 1000.0
        importance=1.0
    )

    results = await archive.retrieve(query_event)

    assert len(results) == 2
    # the newer event (timestamp 950) should suffer less decay penalty and rank first
    assert results[0]['id'] == str(event_new.id)
    assert results[1]['id'] == str(event_old.id)

@pytest.mark.asyncio
async def test_purge_events(mock_config):
    mock_config.CURRENT_TIME = 10000.0

    events = [
        # old, low importance -> SHOULD PURGE
        GameEventModel(id=uuid.uuid4(), label="Boring old", timestamp=100.0, importance=1.0),
        # old, high importance -> KEEP (Threshold is 5.0)
        GameEventModel(id=uuid.uuid4(), label="Important old", timestamp=100.0, importance=8.0),
        # recent, low importance -> KEEP (Delta time < 60 mins)
        GameEventModel(id=uuid.uuid4(), label="Boring new", timestamp=9900.0, importance=1.0)
    ]
    await archive.embed_bunch(events)

    await archive.purge_events()

    # verify
    remaining = archive._collection.get()
    assert len(remaining["ids"]) == 2
    remaining_docs = remaining["documents"]
    assert "Boring old" not in remaining_docs
    assert "Important old" in remaining_docs
    assert "Boring new" in remaining_docs


@pytest.mark.asyncio
async def test_purge_boundary_conditions(mock_config):
    """test events sitting exactly on the purge thresholds."""
    mock_config.CURRENT_TIME = 1000.0
    mock_config.EVENT_PURGE_TIME = 60.0
    mock_config.EVENT_PURGE_IMPORTANCE_THRESHOLD = 5.0

    threshold_time = mock_config.CURRENT_TIME - mock_config.EVENT_PURGE_TIME

    events = [
        # exactly on the time threshold, low importance
        # age > 60 AND importance < 5.0 -> PURGE.
        GameEventModel(id=uuid.uuid4(), label="Boundary Time", timestamp=threshold_time, importance=1.0),

        # old, but exactly on the importance threshold -> KEEP
        GameEventModel(id=uuid.uuid4(), label="Boundary Importance", timestamp=100.0, importance=5.0),
    ]

    await archive.embed_bunch(events)
    await archive.purge_events()

    remaining = archive._collection.get()
    remaining_docs = remaining["documents"]

    # Adjust these assertions based on whether your operators are strict (<) or inclusive (<=)
    assert "Boundary Importance" in remaining_docs

@pytest.mark.asyncio
async def test_delete_events_by_id():
    """ensure specific events can be deleted by their IDs while leaving others intact."""
    event_to_delete_1 = GameEventModel(
        id=uuid.uuid4(), label="Target 1", timestamp=100.0, importance=1.0
    )
    event_to_delete_2 = GameEventModel(
        id=uuid.uuid4(), label="Target 2", timestamp=200.0, importance=1.0
    )
    event_to_keep = GameEventModel(
        id=uuid.uuid4(), label="Safe Event", timestamp=300.0, importance=5.0
    )

    await archive.embed_bunch([event_to_delete_1, event_to_delete_2, event_to_keep])

    # verify all three were inserted
    initial_results = archive._collection.get()
    assert len(initial_results["ids"]) == 3

    # delete the first two events by their IDs
    ids_to_delete = [str(event_to_delete_1.id), str(event_to_delete_2.id)]
    await archive.delete_events_by_id(ids_to_delete)

    # only the third event should remain
    remaining = archive._collection.get()
    assert len(remaining["ids"]) == 1
    assert remaining["ids"][0] == str(event_to_keep.id)
    assert remaining["documents"][0] == "Safe Event"