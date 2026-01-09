import time

import pytest

from providers.health_monitor_provider import (
    HealthMonitorProvider,
    HealthStatus,
    ProviderHealth,
)


@pytest.fixture
def reset_singleton():
    """Reset singleton instances between tests."""
    HealthMonitorProvider.reset()  # type: ignore
    yield
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def health_monitor(reset_singleton):
    """Create a fresh HealthMonitorProvider for each test."""
    monitor = HealthMonitorProvider(heartbeat_timeout=5.0, error_threshold=3)
    return monitor


class TestProviderHealth:
    """Tests for ProviderHealth dataclass."""

    def test_provider_health_to_dict(self):
        """Test conversion to dictionary."""
        health = ProviderHealth(
            name="test_provider",
            status=HealthStatus.HEALTHY,
            last_heartbeat=time.time(),
            error_count=2,
            last_error="Test error",
            metadata={"version": "1.0"},
        )

        result = health.to_dict()

        assert result["name"] == "test_provider"
        assert result["status"] == "healthy"
        assert result["error_count"] == 2
        assert result["last_error"] == "Test error"
        assert result["metadata"]["version"] == "1.0"
        assert "seconds_since_heartbeat" in result


class TestHealthMonitorProvider:
    """Tests for HealthMonitorProvider."""

    def test_register_provider(self, health_monitor):
        """Test provider registration."""
        health_monitor.register("camera", metadata={"type": "USB"})

        status = health_monitor.get_provider_status("camera")
        assert status is not None
        assert status.name == "camera"
        assert status.metadata["type"] == "USB"

    def test_heartbeat_updates_status(self, health_monitor):
        """Test that heartbeat updates provider status to healthy."""
        health_monitor.register("lidar")
        health_monitor.heartbeat("lidar")

        status = health_monitor.get_provider_status("lidar")
        assert status.status == HealthStatus.HEALTHY
        assert status.last_heartbeat is not None

    def test_heartbeat_auto_registers(self, health_monitor):
        """Test that heartbeat auto-registers unknown providers."""
        health_monitor.heartbeat("new_provider")

        status = health_monitor.get_provider_status("new_provider")
        assert status is not None
        assert status.status == HealthStatus.HEALTHY

    def test_report_error_increments_count(self, health_monitor):
        """Test that reporting errors increments error count."""
        health_monitor.register("audio")
        health_monitor.report_error("audio", "Connection lost")
        health_monitor.report_error("audio", "Timeout")

        status = health_monitor.get_provider_status("audio")
        assert status.error_count == 2
        assert status.last_error == "Timeout"

    def test_error_threshold_changes_status(self, health_monitor):
        """Test that exceeding error threshold changes status to degraded."""
        health_monitor.register("gps")

        for i in range(3):
            health_monitor.report_error("gps", f"Error {i}")

        status = health_monitor.get_provider_status("gps")
        assert status.status == HealthStatus.DEGRADED

    def test_reset_errors(self, health_monitor):
        """Test resetting error count."""
        health_monitor.register("motor")
        health_monitor.report_error("motor", "Overheating")
        health_monitor.reset_errors("motor")

        status = health_monitor.get_provider_status("motor")
        assert status.error_count == 0
        assert status.last_error is None

    def test_unregister_provider(self, health_monitor):
        """Test unregistering a provider."""
        health_monitor.register("temp_sensor")
        health_monitor.unregister("temp_sensor")

        status = health_monitor.get_provider_status("temp_sensor")
        assert status is None

    def test_get_all_status(self, health_monitor):
        """Test getting all provider statuses."""
        health_monitor.register("provider1")
        health_monitor.register("provider2")
        health_monitor.heartbeat("provider1")

        all_status = health_monitor.get_all_status()

        assert "provider1" in all_status
        assert "provider2" in all_status
        assert all_status["provider1"]["status"] == "healthy"

    def test_is_healthy_all_healthy(self, health_monitor):
        """Test is_healthy when all providers are healthy."""
        health_monitor.register("p1")
        health_monitor.register("p2")
        health_monitor.heartbeat("p1")
        health_monitor.heartbeat("p2")

        assert health_monitor.is_healthy() is True

    def test_is_healthy_with_unhealthy(self, health_monitor):
        """Test is_healthy when a provider is unhealthy."""
        health_monitor.register("healthy_provider")
        health_monitor.heartbeat("healthy_provider")

        health_monitor.register("unhealthy_provider")
        status = health_monitor.get_provider_status("unhealthy_provider")
        status.status = HealthStatus.UNHEALTHY

        assert health_monitor.is_healthy() is False

    def test_heartbeat_timeout(self, health_monitor):
        """Test that providers become unhealthy after timeout."""
        health_monitor.register("timeout_provider")
        health_monitor.heartbeat("timeout_provider")

        status = health_monitor.get_provider_status("timeout_provider")
        status.last_heartbeat = time.time() - 10  # Simulate old heartbeat

        all_status = health_monitor.get_all_status()
        assert all_status["timeout_provider"]["status"] == "unhealthy"

    def test_get_unhealthy_providers(self, health_monitor):
        """Test getting list of unhealthy providers."""
        health_monitor.register("healthy")
        health_monitor.heartbeat("healthy")

        health_monitor.register("unhealthy")
        status = health_monitor.get_provider_status("unhealthy")
        status.status = HealthStatus.UNHEALTHY

        unhealthy = health_monitor.get_unhealthy_providers()
        assert "unhealthy" in unhealthy
        assert "healthy" not in unhealthy

    def test_get_summary(self, health_monitor):
        """Test getting health summary."""
        health_monitor.register("p1")
        health_monitor.register("p2")
        health_monitor.heartbeat("p1")

        summary = health_monitor.get_summary()

        assert "overall_status" in summary
        assert "total_providers" in summary
        assert summary["total_providers"] == 2
        assert summary["healthy"] == 1
        assert "uptime_seconds" in summary

    def test_summary_overall_status_unhealthy(self, health_monitor):
        """Test that summary shows unhealthy when any provider is unhealthy."""
        health_monitor.register("good")
        health_monitor.heartbeat("good")

        health_monitor.register("bad")
        status = health_monitor.get_provider_status("bad")
        status.status = HealthStatus.UNHEALTHY

        summary = health_monitor.get_summary()
        assert summary["overall_status"] == "unhealthy"

    def test_heartbeat_with_metadata(self, health_monitor):
        """Test heartbeat with metadata update."""
        health_monitor.register("sensor")
        health_monitor.heartbeat("sensor", metadata={"temperature": 25.5})

        status = health_monitor.get_provider_status("sensor")
        assert status.metadata["temperature"] == 25.5
