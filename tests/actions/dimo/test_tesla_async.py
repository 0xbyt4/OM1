# tests/actions/dimo/test_tesla_async.py

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock DIMO dependency before importing
mock_dimo_module = MagicMock()
mock_dimo_module.DIMO = MagicMock()
sys.modules["dimo"] = mock_dimo_module

from actions.dimo.connector.tesla import (  # noqa: E402
    DIMOTeslaConfig,
    DIMOTeslaConnector,
)
from actions.dimo.interface import TeslaInput  # noqa: E402


class TestDIMOTeslaAsyncBehavior:
    """Unit tests for DIMOTeslaConnector async HTTP behavior."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return DIMOTeslaConfig(
            client_id="test_client",
            domain="test_domain",
            private_key="test_key",
            token_id=12345,
        )

    @pytest.fixture
    def connector(self, mock_config):
        """Create DIMOTeslaConnector with mocked dependencies."""
        with patch("actions.dimo.connector.tesla.DIMO") as mock_dimo:
            mock_dimo_instance = MagicMock()
            mock_dimo_instance.auth.get_dev_jwt.return_value = {"access_token": "test"}
            mock_dimo_instance.token_exchange.exchange.return_value = {"token": "test"}
            mock_dimo.return_value = mock_dimo_instance

            with patch("actions.dimo.connector.tesla.IOProvider"):
                connector = DIMOTeslaConnector(mock_config)
                connector.vehicle_jwt = "test_jwt"
                connector.token_id = 12345
                return connector

    def create_aiohttp_mocks(self, status=200):
        """Create properly configured aiohttp mocks."""
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.text = AsyncMock(return_value="OK")

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        return mock_session_cm, mock_session

    @pytest.mark.asyncio
    async def test_connect_lock_doors_uses_aiohttp(self, connector):
        """connect() lock doors should use aiohttp for non-blocking HTTP."""
        mock_session_cm, mock_session = self.create_aiohttp_mocks()

        with (
            patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
            patch(
                "actions.dimo.connector.tesla.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            output = TeslaInput(action="lock doors")
            await connector.connect(output)

            mock_session.post.assert_called_once()
            assert "doors/lock" in mock_session.post.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connect_unlock_doors_uses_aiohttp(self, connector):
        """connect() unlock doors should use aiohttp for non-blocking HTTP."""
        mock_session_cm, mock_session = self.create_aiohttp_mocks()

        with (
            patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
            patch(
                "actions.dimo.connector.tesla.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            output = TeslaInput(action="unlock doors")
            await connector.connect(output)

            mock_session.post.assert_called_once()
            assert "doors/unlock" in mock_session.post.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connect_open_frunk_uses_aiohttp(self, connector):
        """connect() open frunk should use aiohttp for non-blocking HTTP."""
        mock_session_cm, mock_session = self.create_aiohttp_mocks()

        with (
            patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
            patch(
                "actions.dimo.connector.tesla.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            output = TeslaInput(action="open frunk")
            await connector.connect(output)

            mock_session.post.assert_called_once()
            assert "frunk/open" in mock_session.post.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connect_open_trunk_uses_aiohttp(self, connector):
        """connect() open trunk should use aiohttp for non-blocking HTTP."""
        mock_session_cm, mock_session = self.create_aiohttp_mocks()

        with (
            patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
            patch(
                "actions.dimo.connector.tesla.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            output = TeslaInput(action="open trunk")
            await connector.connect(output)

            mock_session.post.assert_called_once()
            assert "trunk/open" in mock_session.post.call_args[0][0]
