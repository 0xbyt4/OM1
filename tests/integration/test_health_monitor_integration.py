"""
Integration tests for HealthMonitorProvider with OM1 runtime simulation.

These tests verify that health monitoring works correctly in realistic
scenarios that simulate actual OM1 operation.
"""

import logging
import time

import pytest

from providers.health_monitor_provider import (
    HealthMonitorProvider,
    HealthStatus,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    HealthMonitorProvider.reset()  # type: ignore
    yield
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def health_monitor():
    """Create a fresh HealthMonitorProvider with short intervals for testing."""
    return HealthMonitorProvider(
        heartbeat_timeout=2.0,
        error_threshold=3,
        check_interval=1.0,
    )


class TestRuntimeIntegration:
    """Tests simulating CortexRuntime lifecycle."""

    def test_runtime_starts_and_stops_monitoring(self, health_monitor):
        """Verify monitoring starts on runtime start and stops on cleanup."""
        # Simulate CortexRuntime.run() start
        health_monitor.start_monitoring()
        assert health_monitor._monitor_running is True
        assert health_monitor._monitor_thread is not None
        assert health_monitor._monitor_thread.is_alive()

        # Simulate CortexRuntime._cleanup_tasks()
        health_monitor.stop_monitoring()
        assert health_monitor._monitor_running is False

    def test_multiple_start_calls_are_safe(self, health_monitor):
        """Verify calling start_monitoring multiple times is safe."""
        health_monitor.start_monitoring()
        thread1 = health_monitor._monitor_thread

        # Second call should not create new thread
        health_monitor.start_monitoring()
        thread2 = health_monitor._monitor_thread

        assert thread1 is thread2
        health_monitor.stop_monitoring()

    def test_stop_without_start_is_safe(self, health_monitor):
        """Verify calling stop_monitoring without start is safe."""
        # Should not raise
        health_monitor.stop_monitoring()
        assert health_monitor._monitor_running is False


class TestProviderLifecycle:
    """Tests simulating provider registration and heartbeat patterns."""

    def test_provider_auto_registration_on_init(self, health_monitor):
        """Simulate providers registering themselves on __init__."""
        # Simulate what ElevenLabsTTSProvider.__init__ does
        health_monitor.register("ElevenLabsTTSProvider", metadata={"type": "tts"})

        # Simulate what ASRProvider.__init__ does
        health_monitor.register("ASRProvider", metadata={"type": "speech"})

        # Simulate what VLMOpenAIProvider.__init__ does
        health_monitor.register(
            "VLMOpenAIProvider", metadata={"type": "vision", "model": "gpt-4o"}
        )

        summary = health_monitor.get_summary()
        assert summary["total_providers"] == 3

    def test_heartbeat_on_successful_operation(self, health_monitor):
        """Simulate providers sending heartbeat on successful operations."""
        health_monitor.register("TTSProvider")

        # Simulate TTS.add_pending_message() success
        health_monitor.heartbeat("TTSProvider")

        status = health_monitor.get_provider_status("TTSProvider")
        assert status.status == HealthStatus.HEALTHY

    def test_error_reporting_on_api_failure(self, health_monitor):
        """Simulate providers reporting errors on API failures."""
        health_monitor.register("VLMOpenAIProvider")
        health_monitor.heartbeat("VLMOpenAIProvider")

        # Simulate API errors
        health_monitor.report_error("VLMOpenAIProvider", "API rate limit exceeded")
        health_monitor.report_error("VLMOpenAIProvider", "Connection timeout")
        health_monitor.report_error("VLMOpenAIProvider", "Internal server error")

        status = health_monitor.get_provider_status("VLMOpenAIProvider")
        assert status.error_count == 3
        assert status.status == HealthStatus.DEGRADED


class TestInputOrchestratorIntegration:
    """Tests simulating InputOrchestrator behavior."""

    def test_sensors_registered_on_orchestrator_init(self, health_monitor):
        """Simulate InputOrchestrator registering all sensors on init."""
        # Simulate InputOrchestrator.__init__ registering sensors
        sensors = ["CameraSensor", "MicrophoneSensor", "GPSSensor", "LidarSensor"]

        for sensor in sensors:
            health_monitor.register(sensor, metadata={"type": "input_sensor"})

        summary = health_monitor.get_summary()
        assert summary["total_providers"] == 4

    def test_heartbeat_on_sensor_event(self, health_monitor):
        """Simulate heartbeat sent after each sensor event."""
        health_monitor.register("CameraSensor")

        # Simulate multiple camera frames processed
        for _ in range(10):
            health_monitor.heartbeat("CameraSensor")

        status = health_monitor.get_provider_status("CameraSensor")
        assert status.status == HealthStatus.HEALTHY

    def test_error_on_sensor_exception(self, health_monitor):
        """Simulate error reported when sensor raises exception."""
        health_monitor.register("GPSSensor")
        health_monitor.heartbeat("GPSSensor")

        # Simulate GPS disconnection
        health_monitor.report_error("GPSSensor", "Serial port disconnected")

        status = health_monitor.get_provider_status("GPSSensor")
        assert status.error_count == 1
        assert status.last_error == "Serial port disconnected"


class TestRealisticScenarios:
    """Tests simulating real-world failure scenarios."""

    def test_provider_timeout_detection(self, health_monitor):
        """Simulate provider stopping heartbeats and being detected as unhealthy."""
        health_monitor.register("GpsProvider")
        health_monitor.heartbeat("GpsProvider")

        # Simulate GPS stopping heartbeats
        status = health_monitor.get_provider_status("GpsProvider")
        status.last_heartbeat = time.time() - 10  # 10 seconds ago

        # Check if detected as unhealthy
        unhealthy = health_monitor.get_unhealthy_providers()
        assert "GpsProvider" in unhealthy

    def test_graceful_degradation_scenario(self, health_monitor):
        """Simulate system gracefully degrading when provider has issues."""
        # Start with all healthy
        providers = ["VLMOpenAI", "VLMGemini", "ElevenLabsTTS", "GPS"]
        for p in providers:
            health_monitor.register(p)
            health_monitor.heartbeat(p)

        assert health_monitor.is_healthy() is True

        # VLMOpenAI starts having errors
        for i in range(3):
            health_monitor.report_error("VLMOpenAI", f"Error {i}")

        # System should still be "healthy" (degraded != unhealthy)
        summary = health_monitor.get_summary()
        assert summary["degraded"] == 1
        assert summary["healthy"] == 3

    def test_recovery_after_issues(self, health_monitor):
        """Simulate provider recovering after issues."""
        health_monitor.register("TTSProvider")

        # Provider has errors
        for i in range(5):
            health_monitor.report_error("TTSProvider", f"Error {i}")

        status = health_monitor.get_provider_status("TTSProvider")
        assert status.status == HealthStatus.DEGRADED

        # Provider recovers
        health_monitor.reset_errors("TTSProvider")
        health_monitor.heartbeat("TTSProvider")

        status = health_monitor.get_provider_status("TTSProvider")
        assert status.status == HealthStatus.HEALTHY
        assert status.error_count == 0

    def test_multiple_providers_failing(self, health_monitor):
        """Simulate multiple providers failing simultaneously."""
        providers = ["Camera", "GPS", "TTS", "VLM"]
        for p in providers:
            health_monitor.register(p)
            health_monitor.heartbeat(p)

        # Camera and GPS stop responding
        camera_status = health_monitor.get_provider_status("Camera")
        camera_status.last_heartbeat = time.time() - 10

        gps_status = health_monitor.get_provider_status("GPS")
        gps_status.last_heartbeat = time.time() - 15

        # TTS has errors
        for i in range(3):
            health_monitor.report_error("TTS", f"Error {i}")

        summary = health_monitor.get_summary()
        assert summary["unhealthy"] == 2  # Camera, GPS
        assert summary["degraded"] == 1  # TTS
        assert summary["healthy"] == 1  # VLM


class TestBackgroundMonitoringLogs:
    """Tests verifying background monitoring logs correctly."""

    def test_logs_unhealthy_providers(self, health_monitor, caplog):
        """Verify ERROR log when providers are unhealthy."""
        health_monitor.register("FailingProvider")
        health_monitor.heartbeat("FailingProvider")

        # Make it unhealthy
        status = health_monitor.get_provider_status("FailingProvider")
        status.last_heartbeat = time.time() - 10

        with caplog.at_level(logging.ERROR):
            health_monitor._check_and_log_health()

        assert "Unhealthy providers" in caplog.text
        assert "FailingProvider" in caplog.text

    def test_logs_degraded_providers(self, health_monitor, caplog):
        """Verify WARNING log when providers are degraded."""
        health_monitor.register("DegradedProvider")
        health_monitor.heartbeat("DegradedProvider")

        # Make it degraded
        for i in range(3):
            health_monitor.report_error("DegradedProvider", f"Error {i}")

        with caplog.at_level(logging.WARNING):
            health_monitor._check_and_log_health()

        assert "Degraded providers" in caplog.text
        assert "DegradedProvider" in caplog.text

    def test_no_log_when_all_healthy(self, health_monitor, caplog):
        """Verify no logs when all providers are healthy."""
        providers = ["Provider1", "Provider2", "Provider3"]
        for p in providers:
            health_monitor.register(p)
            health_monitor.heartbeat(p)

        with caplog.at_level(logging.WARNING):
            health_monitor._check_and_log_health()

        assert "Unhealthy" not in caplog.text
        assert "Degraded" not in caplog.text

    def test_background_thread_detects_issues(self, health_monitor, caplog):
        """Verify background thread actually runs and detects issues."""
        health_monitor.register("TestProvider")
        health_monitor.heartbeat("TestProvider")

        # Make it unhealthy
        status = health_monitor.get_provider_status("TestProvider")
        status.last_heartbeat = time.time() - 10

        # Start monitoring with short interval
        health_monitor._check_interval = 0.5
        health_monitor.start_monitoring()

        # Wait for at least one check cycle
        with caplog.at_level(logging.ERROR):
            time.sleep(1.0)

        health_monitor.stop_monitoring()

        assert "Unhealthy providers" in caplog.text


class TestSingletonBehavior:
    """Tests verifying singleton works correctly across components."""

    def test_same_instance_across_calls(self):
        """Verify same instance returned across multiple calls."""
        health1 = HealthMonitorProvider()
        health2 = HealthMonitorProvider()

        assert health1 is health2

    def test_providers_visible_across_instances(self):
        """Verify providers registered in one place visible elsewhere."""
        # Simulate provider registering itself
        provider_health = HealthMonitorProvider()
        provider_health.register("MyProvider")
        provider_health.heartbeat("MyProvider")

        # Simulate runtime checking health
        runtime_health = HealthMonitorProvider()
        status = runtime_health.get_provider_status("MyProvider")

        assert status is not None
        assert status.status == HealthStatus.HEALTHY

    def test_monitoring_state_shared(self):
        """Verify monitoring state shared across instances."""
        health1 = HealthMonitorProvider()
        health1.start_monitoring()

        health2 = HealthMonitorProvider()
        assert health2._monitor_running is True

        health2.stop_monitoring()
        assert health1._monitor_running is False


class TestAutoRecoveryIntegration:
    """Integration tests for auto-recovery functionality."""

    def test_recovery_triggered_on_timeout(self, health_monitor):
        """Test that recovery is triggered when provider times out."""
        recovery_calls = []

        def mock_recovery():
            recovery_calls.append(time.time())
            return True

        health_monitor.register(
            "TimeoutProvider",
            recovery_callback=mock_recovery,
        )
        health_monitor.heartbeat("TimeoutProvider")

        # Simulate timeout
        status = health_monitor.get_provider_status("TimeoutProvider")
        status.last_heartbeat = time.time() - 10

        # Trigger check
        health_monitor._check_and_log_health()

        assert len(recovery_calls) == 1

    def test_recovery_not_triggered_when_healthy(self, health_monitor):
        """Test that recovery is not triggered for healthy providers."""
        recovery_calls = []

        def mock_recovery():
            recovery_calls.append(time.time())
            return True

        health_monitor.register(
            "HealthyProvider",
            recovery_callback=mock_recovery,
        )
        health_monitor.heartbeat("HealthyProvider")

        # Trigger check (provider is healthy)
        health_monitor._check_and_log_health()

        assert len(recovery_calls) == 0

    def test_successful_recovery_resets_provider(self, health_monitor):
        """Test that successful recovery resets provider to healthy."""

        def mock_recovery():
            # Simulate successful restart
            health_monitor.heartbeat("RecoveringProvider")
            return True

        health_monitor.register(
            "RecoveringProvider",
            recovery_callback=mock_recovery,
        )
        health_monitor.heartbeat("RecoveringProvider")

        # Simulate timeout
        status = health_monitor.get_provider_status("RecoveringProvider")
        status.last_heartbeat = time.time() - 10

        # Trigger recovery
        health_monitor._check_and_log_health()

        # Provider should be healthy after recovery
        status = health_monitor.get_provider_status("RecoveringProvider")
        assert status.status == HealthStatus.HEALTHY

    def test_failed_recovery_increments_attempts(self, health_monitor):
        """Test that failed recovery increments attempt counter."""

        def mock_recovery():
            return False

        health_monitor.register(
            "FailingProvider",
            recovery_callback=mock_recovery,
        )
        health_monitor._recovery_cooldown = 0

        # Multiple failed recovery attempts
        for _ in range(3):
            health_monitor._attempt_recovery("FailingProvider")

        recovery_status = health_monitor.get_recovery_status("FailingProvider")
        assert recovery_status["attempts"] == 3

    def test_recovery_respects_max_attempts(self, health_monitor):
        """Test that recovery stops after max attempts."""
        recovery_calls = []

        def mock_recovery():
            recovery_calls.append(1)
            return False

        health_monitor.register(
            "MaxAttemptsProvider",
            recovery_callback=mock_recovery,
        )
        health_monitor._recovery_cooldown = 0

        # Try more than max attempts
        for _ in range(5):
            health_monitor._attempt_recovery("MaxAttemptsProvider")

        # Should only have called recovery 3 times (default max)
        assert len(recovery_calls) == 3

    def test_multiple_providers_recovery(self, health_monitor):
        """Test recovery of multiple providers simultaneously."""
        recovery_order = []

        def make_recovery(name):
            def recovery():
                recovery_order.append(name)
                return True

            return recovery

        # Register multiple providers with recovery
        providers = ["Provider1", "Provider2", "Provider3"]
        for name in providers:
            health_monitor.register(name, recovery_callback=make_recovery(name))
            health_monitor.heartbeat(name)

        # Make all unhealthy
        for name in providers:
            status = health_monitor.get_provider_status(name)
            status.last_heartbeat = time.time() - 10

        # Trigger check
        health_monitor._check_and_log_health()

        # All should have been recovered
        assert len(recovery_order) == 3
        assert set(recovery_order) == set(providers)

    def test_recovery_with_error_reset(self, health_monitor):
        """Test recovery that resets error state."""

        def mock_recovery():
            health_monitor.reset_errors("ErrorProvider")
            health_monitor.heartbeat("ErrorProvider")
            return True

        health_monitor.register(
            "ErrorProvider",
            recovery_callback=mock_recovery,
        )

        # Generate errors
        for i in range(5):
            health_monitor.report_error("ErrorProvider", f"Error {i}")

        status = health_monitor.get_provider_status("ErrorProvider")
        assert status.status == HealthStatus.DEGRADED

        # Trigger recovery
        health_monitor._attempt_recovery("ErrorProvider")

        # Should be healthy with no errors
        status = health_monitor.get_provider_status("ErrorProvider")
        assert status.status == HealthStatus.HEALTHY
        assert status.error_count == 0
