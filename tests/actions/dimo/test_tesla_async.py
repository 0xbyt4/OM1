# tests/actions/dimo/test_tesla_async.py

import sys
from unittest.mock import MagicMock, patch

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

    @pytest.mark.asyncio
    async def test_connect_lock_doors_uses_asyncio_to_thread(self, connector):
        """connect() lock doors should use asyncio.to_thread for non-blocking HTTP."""
        with patch("actions.dimo.connector.tesla.asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            output = TeslaInput(action="lock doors")
            await connector.connect(output)

            assert mock_to_thread.called

    @pytest.mark.asyncio
    async def test_connect_unlock_doors_uses_asyncio_to_thread(self, connector):
        """connect() unlock doors should use asyncio.to_thread for non-blocking HTTP."""
        with patch("actions.dimo.connector.tesla.asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            output = TeslaInput(action="unlock doors")
            await connector.connect(output)

            assert mock_to_thread.called

    @pytest.mark.asyncio
    async def test_connect_open_frunk_uses_asyncio_to_thread(self, connector):
        """connect() open frunk should use asyncio.to_thread for non-blocking HTTP."""
        with patch("actions.dimo.connector.tesla.asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            output = TeslaInput(action="open frunk")
            await connector.connect(output)

            assert mock_to_thread.called

    @pytest.mark.asyncio
    async def test_connect_open_trunk_uses_asyncio_to_thread(self, connector):
        """connect() open trunk should use asyncio.to_thread for non-blocking HTTP."""
        with patch("actions.dimo.connector.tesla.asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            output = TeslaInput(action="open trunk")
            await connector.connect(output)

            assert mock_to_thread.called
