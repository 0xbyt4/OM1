# tests/inputs/plugins/test_fabric_closest_peer_async.py

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.fabric_closest_peer import (
    FabricClosestPeer,
    FabricClosestPeerConfig,
)


class TestFabricClosestPeerAsyncBehavior:
    """Unit tests for FabricClosestPeer async HTTP behavior."""

    @pytest.fixture
    def config(self):
        """Create config with mock_mode disabled."""
        return FabricClosestPeerConfig(
            input_name="test",
            fabric_endpoint="http://test.endpoint",
            mock_mode=False,
        )

    @pytest.fixture
    def peer_instance(self, config):
        """Create FabricClosestPeer instance."""
        return FabricClosestPeer(config)

    @pytest.mark.asyncio
    async def test_poll_uses_asyncio_to_thread_for_http(self, peer_instance):
        """_poll() should use asyncio.to_thread for non-blocking HTTP."""
        with patch(
            "inputs.plugins.fabric_closest_peer.asyncio.to_thread"
        ) as mock_to_thread:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]
            }
            mock_to_thread.return_value = mock_response

            with patch.object(peer_instance.io, "get_dynamic_variable") as mock_gps:
                mock_gps.side_effect = lambda key: 40.0 if key == "latitude" else -74.0

                await peer_instance._poll()

                assert mock_to_thread.called

    @pytest.mark.asyncio
    async def test_poll_does_not_block_event_loop(self, peer_instance):
        """_poll() should not block the event loop during HTTP requests."""
        quick_task_started = asyncio.Event()
        quick_task_finished = asyncio.Event()

        async def quick_task():
            quick_task_started.set()
            await asyncio.sleep(0.01)
            quick_task_finished.set()

        with patch("inputs.plugins.fabric_closest_peer.requests.post") as mock_post:

            def slow_post(*args, **kwargs):
                import time

                time.sleep(0.5)
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]
                }
                return mock_response

            mock_post.side_effect = slow_post

            with patch.object(peer_instance.io, "get_dynamic_variable") as mock_gps:
                mock_gps.side_effect = lambda key: 40.0 if key == "latitude" else -74.0

                poll_task = asyncio.create_task(peer_instance._poll())
                quick_task_coro = asyncio.create_task(quick_task())

                await asyncio.sleep(0.1)

                if not quick_task_started.is_set():
                    poll_task.cancel()
                    quick_task_coro.cancel()
                    try:
                        await poll_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await quick_task_coro
                    except asyncio.CancelledError:
                        pass
                    pytest.fail("Event loop was blocked during _poll()")

                try:
                    await asyncio.wait_for(poll_task, timeout=2.0)
                except asyncio.TimeoutError:
                    poll_task.cancel()

                await quick_task_coro
