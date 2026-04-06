import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from api.schemas import GameEventModel
from core.orchestrator_service import JobType, OrchestratorService, Signal, SignalType


@pytest.mark.asyncio
async def test_publish_enqueues_signal():
    service = OrchestratorService()
    signal = Signal(type=SignalType.IDLE_INCREMENT, value=1)

    await service.publish(signal)

    queued = await service._event_queue.get()
    assert queued == signal


@pytest.mark.asyncio
async def test_get_last_event_uses_timestamp_specific_lookup():
    service = OrchestratorService()
    event = GameEventModel(label="Signal", timestamp=42.0)

    with patch("core.orchestrator_service.state_buffer.get_last_event", new_callable=AsyncMock) as get_last, \
         patch("core.orchestrator_service.state_buffer.get_last_event_before", new_callable=AsyncMock) as get_before:
        get_last.return_value = event
        get_before.return_value = event

        result_without_cutoff = await service._get_last_event(None)
        result_with_cutoff = await service._get_last_event(100.0)

        assert result_without_cutoff == event
        assert result_with_cutoff == event
        get_last.assert_awaited_once()
        get_before.assert_awaited_once_with(100.0)


@pytest.mark.asyncio
async def test_reducer_force_director_enqueues_with_latest_event():
    service = OrchestratorService()
    trigger_event = GameEventModel(label="Door opens", timestamp=100.0)

    with patch.object(service, "_get_last_event", new_callable=AsyncMock) as get_last, \
         patch.object(service, "_enqueue_once", new_callable=AsyncMock) as enqueue_once:
        get_last.return_value = trigger_event

        loop_task = asyncio.create_task(service._reducer_loop())
        await service.publish(Signal(type=SignalType.FORCE_DIRECTOR, value=0, timestamp=100.0))

        # Let the reducer consume one signal.
        await asyncio.sleep(0.05)
        service._stop.set()
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)

        get_last.assert_awaited_with(100.0)
        enqueue_once.assert_awaited_with(JobType.RUN_DIRECTOR, director_payload=trigger_event)


@pytest.mark.asyncio
async def test_reducer_idle_threshold_triggers_director_and_resets_idle_sum():
    service = OrchestratorService()
    trigger_event = GameEventModel(label="Radar ping", timestamp=300.0)

    with patch.object(service, "_get_last_event", new_callable=AsyncMock) as get_last, \
         patch.object(service, "_enqueue_once", new_callable=AsyncMock) as enqueue_once:
        get_last.return_value = trigger_event

        loop_task = asyncio.create_task(service._reducer_loop())
        await service.publish(Signal(type=SignalType.IDLE_INCREMENT, value=10, timestamp=300.0))

        await asyncio.sleep(0.05)
        service._stop.set()
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)

        enqueue_once.assert_awaited_with(JobType.RUN_DIRECTOR, director_payload=trigger_event)
        assert service.state.idle_sum == 0


@pytest.mark.asyncio
async def test_scheduler_enqueues_pruner_using_in_game_time():
    service = OrchestratorService()
    current_event = GameEventModel(label="Current game state", timestamp=1_000_000.0)

    async def fast_sleep(_):
        return None

    async def enqueue_and_stop(job_type, director_payload=None):
        assert job_type == JobType.RUN_PRUNER
        service._stop.set()

    with patch("core.orchestrator_service.asyncio.sleep", side_effect=fast_sleep), \
         patch("core.orchestrator_service.state_buffer.get_last_event", new_callable=AsyncMock) as get_last_event, \
         patch.object(service, "_enqueue_once", side_effect=enqueue_and_stop, new_callable=AsyncMock) as enqueue_once:
        get_last_event.return_value = current_event

        await service._scheduler_loop()

        enqueue_once.assert_awaited_once()
        assert service.state.last_prune_timestamp == current_event.timestamp


@pytest.mark.asyncio
async def test_worker_dispatches_jobs_to_expected_handlers():
    service = OrchestratorService()
    director_event = GameEventModel(label="Player entered lab", timestamp=500.0)

    async def stop_after_miner():
        return None

    async def stop_after_director(event):
        assert event == director_event
        return None

    async def stop_after_pruner():
        service._stop.set()

    with patch("core.orchestrator_service.run_miner", side_effect=stop_after_miner, new_callable=AsyncMock) as run_miner, \
         patch("core.orchestrator_service.run_director", side_effect=stop_after_director, new_callable=AsyncMock) as run_director, \
         patch("core.orchestrator_service.run_pruner", side_effect=stop_after_pruner, new_callable=AsyncMock) as run_pruner:

        await service._job_queue.put((JobType.RUN_MINER, None))
        await service._job_queue.put((JobType.RUN_DIRECTOR, director_event))
        await service._job_queue.put((JobType.RUN_PRUNER, None))

        await service._worker_loop()

        run_miner.assert_awaited_once()
        run_director.assert_awaited_once_with(director_event)
        run_pruner.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_and_stop_manage_background_tasks():
    service = OrchestratorService()

    await service.start()

    assert len(service._tasks) == 3
    assert all(not task.done() for task in service._tasks)

    await service.stop()

    assert service._stop.is_set()
    assert all(task.cancelled() or task.done() for task in service._tasks)

