# tests/inputs/plugins/test_fabric_closest_peer_async.py

from unittest.mock import AsyncMock, MagicMock, patch

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

    def create_aiohttp_mocks(self, response_data):
        """Create properly configured aiohttp mocks."""
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=response_data)

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        return mock_session_cm, mock_session, mock_response

    @pytest.mark.asyncio
    async def test_poll_uses_aiohttp_for_http(self, peer_instance):
        """_poll() should use aiohttp for non-blocking HTTP."""
        response_data = {"result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]}
        mock_session_cm, mock_session, _ = self.create_aiohttp_mocks(response_data)

        with (
            patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
            patch(
                "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            with patch.object(peer_instance.io, "get_dynamic_variable") as mock_gps:
                mock_gps.side_effect = lambda key: 40.0 if key == "latitude" else -74.0

                await peer_instance._poll()

                mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_passes_correct_parameters(self, peer_instance):
        """_poll() should pass correct JSON-RPC parameters to aiohttp."""
        response_data = {"result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]}
        mock_session_cm, mock_session, _ = self.create_aiohttp_mocks(response_data)

        with (
            patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
            patch(
                "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            with patch.object(peer_instance.io, "get_dynamic_variable") as mock_gps:
                mock_gps.side_effect = lambda key: 40.0 if key == "latitude" else -74.0

                await peer_instance._poll()

                call_kwargs = mock_session.post.call_args[1]
                assert call_kwargs["json"]["method"] == "omp2p_findClosestPeer"
                assert call_kwargs["json"]["params"][0]["latitude"] == 40.0
                assert call_kwargs["json"]["params"][0]["longitude"] == -74.0

    @pytest.mark.asyncio
    async def test_poll_returns_peer_coordinates(self, peer_instance):
        """_poll() should return human-readable message with coordinates."""
        response_data = {
            "result": [{"peer": {"latitude": 40.12345, "longitude": -74.98765}}]
        }
        mock_session_cm, mock_session, _ = self.create_aiohttp_mocks(response_data)

        with (
            patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
            patch(
                "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            with patch.object(peer_instance.io, "get_dynamic_variable") as mock_gps:
                mock_gps.side_effect = lambda key: 40.0 if key == "latitude" else -74.0

                result = await peer_instance._poll()

                assert result == "Closest peer at 40.12345, -74.98765"
