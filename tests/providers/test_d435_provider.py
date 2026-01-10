from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    HealthMonitorProvider.reset()  # type: ignore
    # D435Provider reset is done in mock_dependencies
    yield
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.d435_provider.open_zenoh_session") as mock_zenoh_session,
        patch("providers.d435_provider.sensor_msgs") as mock_sensor_msgs,
    ):
        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        yield mock_zenoh_session, mock_session, mock_sensor_msgs


def test_initialization(mock_dependencies):
    """Test D435Provider initialization."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    mock_zenoh_session, mock_session, _ = mock_dependencies
    provider = D435Provider()

    mock_zenoh_session.assert_called_once()
    mock_session.declare_subscriber.assert_called_once()
    assert provider.running is True


def test_singleton_pattern(mock_dependencies):
    """Test singleton pattern."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider1 = D435Provider()
    provider2 = D435Provider()
    assert provider1 is provider2


def test_start(mock_dependencies):
    """Test start method."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider = D435Provider()

    assert provider.running is True


def test_stop(mock_dependencies):
    """Test stop method."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    mock_zenoh_session, mock_session, _ = mock_dependencies
    provider = D435Provider()
    provider.stop()

    assert provider.running is False
    mock_session.close.assert_called_once()


def test_recovery_callback_registered(mock_dependencies):
    """Test that D435Provider registers a recovery callback."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider = D435Provider()
    health = HealthMonitorProvider()

    assert "D435Provider" in health._recovery_callbacks
    assert health._recovery_callbacks["D435Provider"] == provider._recover


def test_recover_calls_stop_and_start(mock_dependencies):
    """Test that _recover stops and restarts the provider."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider = D435Provider()

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(mock_dependencies):
    """Test that _recover returns False when an exception occurs."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider = D435Provider()

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False


def test_calculate_angle_and_distance(mock_dependencies):
    """Test angle and distance calculation."""
    from providers.d435_provider import D435Provider

    D435Provider.reset()  # type: ignore

    provider = D435Provider()

    # Test with known values
    angle, distance = provider.calculate_angle_and_distance(1.0, 0.0)
    assert distance == 1.0
    assert angle == 0.0

    angle, distance = provider.calculate_angle_and_distance(0.0, 1.0)
    assert distance == 1.0
    assert angle == 90.0
