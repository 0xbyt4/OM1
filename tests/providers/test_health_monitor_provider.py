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

    def test_start_stop_monitoring(self, health_monitor):
        """Test starting and stopping background monitoring."""
        health_monitor.start_monitoring()
        assert health_monitor._monitor_running is True
        assert health_monitor._monitor_thread is not None
        assert health_monitor._monitor_thread.is_alive()

        health_monitor.stop_monitoring()
        assert health_monitor._monitor_running is False

    def test_monitoring_logs_unhealthy(self, health_monitor, caplog):
        """Test that monitoring logs unhealthy providers."""
        import logging

        health_monitor.register("test_provider")
        health_monitor.heartbeat("test_provider")

        # Simulate timeout
        status = health_monitor.get_provider_status("test_provider")
        status.last_heartbeat = time.time() - 10  # 10s ago, timeout is 5s

        with caplog.at_level(logging.ERROR):
            health_monitor._check_and_log_health()

        assert "Unhealthy providers" in caplog.text
        assert "test_provider" in caplog.text

    def test_monitoring_logs_degraded(self, health_monitor, caplog):
        """Test that monitoring logs degraded providers."""
        import logging

        health_monitor.register("degraded_provider")
        health_monitor.heartbeat("degraded_provider")

        # Trigger degradation (3 errors with threshold=3)
        for i in range(3):
            health_monitor.report_error("degraded_provider", f"Error {i}")

        with caplog.at_level(logging.WARNING):
            health_monitor._check_and_log_health()

        assert "Degraded providers" in caplog.text
        assert "degraded_provider" in caplog.text

    def test_monitoring_no_log_when_healthy(self, health_monitor, caplog):
        """Test that monitoring does not log when all providers are healthy."""
        import logging

        health_monitor.register("healthy1")
        health_monitor.register("healthy2")
        health_monitor.heartbeat("healthy1")
        health_monitor.heartbeat("healthy2")

        with caplog.at_level(logging.WARNING):
            health_monitor._check_and_log_health()

        assert "Unhealthy providers" not in caplog.text
        assert "Degraded providers" not in caplog.text


class TestAutoRecovery:
    """Tests for auto-recovery functionality."""

    def test_register_with_recovery_callback(self, health_monitor):
        """Test registering provider with recovery callback."""
        callback_called = []

        def recovery():
            callback_called.append(True)
            return True

        health_monitor.register("test_provider", recovery_callback=recovery)

        assert "test_provider" in health_monitor._recovery_callbacks
        assert "test_provider" in health_monitor._recovery_states

    def test_recovery_callback_called_on_unhealthy(self, health_monitor):
        """Test that recovery callback is called when provider is unhealthy."""
        recovery_attempts = []

        def recovery():
            recovery_attempts.append(time.time())
            return True

        health_monitor.register("failing_provider", recovery_callback=recovery)
        health_monitor.heartbeat("failing_provider")

        # Simulate timeout
        status = health_monitor.get_provider_status("failing_provider")
        status.last_heartbeat = time.time() - 10  # 10s ago, timeout is 5s

        health_monitor._check_and_log_health()

        assert len(recovery_attempts) == 1

    def test_recovery_success_resets_attempts(self, health_monitor):
        """Test that successful recovery resets attempt counter."""

        def recovery():
            return True

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor.heartbeat("provider")

        # Simulate timeout and recovery
        status = health_monitor.get_provider_status("provider")
        status.last_heartbeat = time.time() - 10

        health_monitor._attempt_recovery("provider")

        # Check attempts reset after success
        recovery_status = health_monitor.get_recovery_status("provider")
        assert recovery_status["attempts"] == 0

    def test_recovery_failure_increments_attempts(self, health_monitor):
        """Test that failed recovery increments attempt counter."""

        def recovery():
            return False

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor.heartbeat("provider")

        status = health_monitor.get_provider_status("provider")
        status.last_heartbeat = time.time() - 10

        health_monitor._attempt_recovery("provider")

        recovery_status = health_monitor.get_recovery_status("provider")
        assert recovery_status["attempts"] == 1

    def test_max_recovery_attempts(self, health_monitor):
        """Test that recovery stops after max attempts."""
        attempt_count = []

        def recovery():
            attempt_count.append(1)
            return False

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor._recovery_cooldown = 0  # Disable cooldown for test

        # Attempt recovery multiple times
        for _ in range(5):
            health_monitor._attempt_recovery("provider")

        # Should only attempt 3 times (max_recovery_attempts default)
        assert len(attempt_count) == 3

    def test_recovery_cooldown(self, health_monitor):
        """Test that recovery respects cooldown period."""
        attempt_count = []

        def recovery():
            attempt_count.append(time.time())
            return False

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor._recovery_cooldown = 60  # 60 second cooldown

        # First attempt should work
        health_monitor._attempt_recovery("provider")
        # Second attempt should be skipped due to cooldown
        health_monitor._attempt_recovery("provider")

        assert len(attempt_count) == 1

    def test_heartbeat_resets_recovery_state(self, health_monitor):
        """Test that heartbeat resets recovery attempt counter."""

        def recovery():
            return False

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor._recovery_cooldown = 0

        # Make some failed recovery attempts
        health_monitor._attempt_recovery("provider")
        health_monitor._attempt_recovery("provider")

        recovery_status = health_monitor.get_recovery_status("provider")
        assert recovery_status["attempts"] == 2

        # Heartbeat should reset
        health_monitor.heartbeat("provider")

        recovery_status = health_monitor.get_recovery_status("provider")
        assert recovery_status["attempts"] == 0

    def test_recovery_exception_handling(self, health_monitor, caplog):
        """Test that recovery handles callback exceptions gracefully."""
        import logging

        def recovery():
            raise RuntimeError("Recovery failed!")

        health_monitor.register("provider", recovery_callback=recovery)

        with caplog.at_level(logging.ERROR):
            result = health_monitor._attempt_recovery("provider")

        assert result is False
        assert "Recovery error" in caplog.text

    def test_auto_recovery_disabled(self, reset_singleton):
        """Test that auto-recovery can be disabled."""
        monitor = HealthMonitorProvider(
            heartbeat_timeout=5.0, error_threshold=3, auto_recovery=False
        )

        recovery_called = []

        def recovery():
            recovery_called.append(True)
            return True

        monitor.register("provider", recovery_callback=recovery)
        monitor.heartbeat("provider")

        status = monitor.get_provider_status("provider")
        status.last_heartbeat = time.time() - 10

        monitor._check_and_log_health()

        # Recovery should not be called when auto_recovery is disabled
        assert len(recovery_called) == 0

    def test_unregister_cleans_recovery_state(self, health_monitor):
        """Test that unregister cleans up recovery state."""

        def recovery():
            return True

        health_monitor.register("provider", recovery_callback=recovery)

        assert "provider" in health_monitor._recovery_callbacks
        assert "provider" in health_monitor._recovery_states

        health_monitor.unregister("provider")

        assert "provider" not in health_monitor._recovery_callbacks
        assert "provider" not in health_monitor._recovery_states

    def test_get_recovery_status_returns_none_for_no_callback(self, health_monitor):
        """Test get_recovery_status returns None for providers without callback."""
        health_monitor.register("provider")  # No recovery callback

        status = health_monitor.get_recovery_status("provider")
        assert status is None

    def test_recovering_status(self, health_monitor):
        """Test that provider status is set to RECOVERING during recovery."""
        from providers.health_monitor_provider import HealthStatus

        def recovery():
            # Check status during recovery
            status = health_monitor.get_provider_status("provider")
            assert status.status == HealthStatus.RECOVERING
            return True

        health_monitor.register("provider", recovery_callback=recovery)
        health_monitor.heartbeat("provider")

        health_monitor._attempt_recovery("provider")
