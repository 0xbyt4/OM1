import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .singleton import singleton


class HealthStatus(Enum):
    """
    Enumeration for provider health status.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


@dataclass
class RecoveryState:
    """
    Tracks recovery attempts for a provider.

    Parameters
    ----------
    attempts : int
        Number of recovery attempts made.
    last_attempt : float
        Timestamp of last recovery attempt.
    max_attempts : int
        Maximum recovery attempts before giving up.
    """

    attempts: int = 0
    last_attempt: float = 0.0
    max_attempts: int = 3


@dataclass
class ProviderHealth:
    """
    Health information for a registered provider.

    Parameters
    ----------
    name : str
        The name of the provider.
    status : HealthStatus
        Current health status.
    last_heartbeat : float
        Timestamp of last heartbeat.
    error_count : int
        Number of errors since last reset.
    last_error : str, optional
        Most recent error message.
    metadata : dict
        Additional health metadata.
    """

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing health information.
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "seconds_since_heartbeat": (
                round(time.time() - self.last_heartbeat, 2)
                if self.last_heartbeat
                else None
            ),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }


@singleton
class HealthMonitorProvider:
    """
    A thread-safe singleton for monitoring health of all providers.

    This class provides centralized health monitoring for the OM1 system,
    allowing providers to register, send heartbeats, and report errors.
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        error_threshold: int = 5,
        check_interval: float = 10.0,
        max_recovery_attempts: int = 3,
        recovery_cooldown: float = 60.0,
        auto_recovery: bool = True,
    ):
        """
        Initialize the HealthMonitorProvider.

        Parameters
        ----------
        heartbeat_timeout : float
            Seconds after which a provider is considered unhealthy
            if no heartbeat received. Default is 30.0.
        error_threshold : int
            Number of errors before marking provider as degraded.
            Default is 5.
        check_interval : float
            Interval in seconds between health checks when background
            monitoring is enabled. Default is 10.0.
        max_recovery_attempts : int
            Maximum number of recovery attempts before giving up.
            Default is 3.
        recovery_cooldown : float
            Minimum seconds between recovery attempts for same provider.
            Default is 60.0.
        auto_recovery : bool
            Whether to automatically attempt recovery for unhealthy providers.
            Default is True.
        """
        self._lock: threading.Lock = threading.Lock()
        self._providers: Dict[str, ProviderHealth] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._error_threshold = error_threshold
        self._check_interval = check_interval
        self._start_time = time.time()

        # Recovery configuration
        self._max_recovery_attempts = max_recovery_attempts
        self._recovery_cooldown = recovery_cooldown
        self._auto_recovery = auto_recovery
        self._recovery_callbacks: Dict[str, Callable[[], bool]] = {}
        self._recovery_states: Dict[str, RecoveryState] = {}

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False

        logging.info("HealthMonitorProvider initialized")

    def register(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        recovery_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a provider for health monitoring.

        Parameters
        ----------
        name : str
            Unique name for the provider.
        metadata : dict, optional
            Additional metadata about the provider.
        recovery_callback : callable, optional
            A function that attempts to recover the provider.
            Should return True if recovery succeeded, False otherwise.
            The callback should handle its own error catching.
        """
        with self._lock:
            if name in self._providers:
                logging.warning(f"Provider '{name}' already registered, updating")

            self._providers[name] = ProviderHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                last_heartbeat=time.time(),
                metadata=metadata or {},
            )

            if recovery_callback:
                self._recovery_callbacks[name] = recovery_callback
                self._recovery_states[name] = RecoveryState(
                    max_attempts=self._max_recovery_attempts
                )
                logging.debug(
                    f"Provider '{name}' registered with auto-recovery enabled"
                )
            else:
                logging.debug(f"Provider '{name}' registered for health monitoring")

    def unregister(self, name: str) -> None:
        """
        Unregister a provider from health monitoring.

        Parameters
        ----------
        name : str
            Name of the provider to unregister.
        """
        with self._lock:
            if name in self._providers:
                del self._providers[name]
            if name in self._recovery_callbacks:
                del self._recovery_callbacks[name]
            if name in self._recovery_states:
                del self._recovery_states[name]
            logging.debug(f"Provider '{name}' unregistered from health monitoring")

    def heartbeat(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a heartbeat from a provider.

        Parameters
        ----------
        name : str
            Name of the provider sending heartbeat.
        metadata : dict, optional
            Optional metadata to update.
        """
        with self._lock:
            if name not in self._providers:
                self._providers[name] = ProviderHealth(name=name)

            provider = self._providers[name]
            provider.last_heartbeat = time.time()
            provider.status = HealthStatus.HEALTHY

            # Reset recovery state on successful heartbeat
            if name in self._recovery_states:
                self._recovery_states[name].attempts = 0

            if metadata:
                provider.metadata.update(metadata)

    def report_error(self, name: str, error: str) -> None:
        """
        Report an error from a provider.

        Parameters
        ----------
        name : str
            Name of the provider reporting error.
        error : str
            Error message or description.
        """
        with self._lock:
            if name not in self._providers:
                self._providers[name] = ProviderHealth(name=name)

            provider = self._providers[name]
            provider.error_count += 1
            provider.last_error = error

            if provider.error_count >= self._error_threshold:
                provider.status = HealthStatus.DEGRADED

            logging.warning(
                f"Provider '{name}' reported error ({provider.error_count}): {error}"
            )

    def reset_errors(self, name: str) -> None:
        """
        Reset error count for a provider.

        Parameters
        ----------
        name : str
            Name of the provider.
        """
        with self._lock:
            if name in self._providers:
                self._providers[name].error_count = 0
                self._providers[name].last_error = None

    def get_provider_status(self, name: str) -> Optional[ProviderHealth]:
        """
        Get health status for a specific provider.

        Parameters
        ----------
        name : str
            Name of the provider.

        Returns
        -------
        ProviderHealth or None
            Health information for the provider, or None if not found.
        """
        with self._lock:
            return self._providers.get(name)

    def get_all_status(self) -> Dict[str, dict]:
        """
        Get health status for all registered providers.

        Returns
        -------
        dict
            Dictionary mapping provider names to their health status.
        """
        with self._lock:
            current_time = time.time()
            result = {}

            for name, provider in self._providers.items():
                if (
                    provider.last_heartbeat
                    and current_time - provider.last_heartbeat > self._heartbeat_timeout
                ):
                    provider.status = HealthStatus.UNHEALTHY

                result[name] = provider.to_dict()

            return result

    def is_healthy(self) -> bool:
        """
        Check if all providers are healthy.

        Returns
        -------
        bool
            True if all providers are healthy, False otherwise.
        """
        with self._lock:
            current_time = time.time()

            for provider in self._providers.values():
                if provider.status == HealthStatus.UNHEALTHY:
                    return False

                if (
                    provider.last_heartbeat
                    and current_time - provider.last_heartbeat > self._heartbeat_timeout
                ):
                    return False

            return True

    def get_unhealthy_providers(self) -> List[str]:
        """
        Get list of unhealthy provider names.

        Returns
        -------
        list
            Names of providers that are unhealthy or have timed out.
        """
        with self._lock:
            current_time = time.time()
            unhealthy = []

            for name, provider in self._providers.items():
                if provider.status == HealthStatus.UNHEALTHY:
                    unhealthy.append(name)
                elif (
                    provider.last_heartbeat
                    and current_time - provider.last_heartbeat > self._heartbeat_timeout
                ):
                    unhealthy.append(name)

            return unhealthy

    def get_summary(self) -> dict:
        """
        Get a summary of system health.

        Returns
        -------
        dict
            Summary including counts and overall status.
        """
        with self._lock:
            current_time = time.time()
            healthy_count = 0
            degraded_count = 0
            unhealthy_count = 0
            unknown_count = 0

            for provider in self._providers.values():
                if (
                    provider.last_heartbeat
                    and current_time - provider.last_heartbeat > self._heartbeat_timeout
                ):
                    unhealthy_count += 1
                elif provider.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif provider.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                elif provider.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                else:
                    unknown_count += 1

            total = len(self._providers)
            overall_status = HealthStatus.HEALTHY

            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_count > 0 or unknown_count > 0:
                overall_status = HealthStatus.DEGRADED

            return {
                "overall_status": overall_status.value,
                "uptime_seconds": round(current_time - self._start_time, 2),
                "total_providers": total,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unknown": unknown_count,
            }

    def start_monitoring(self) -> None:
        """
        Start background health monitoring thread.

        The monitor periodically checks provider health and logs warnings
        when providers are unhealthy or degraded. Only logs when issues
        are detected to avoid log spam.
        """
        if self._monitor_running:
            logging.warning("Health monitoring is already running")
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="HealthMonitor"
        )
        self._monitor_thread.start()
        logging.info(
            f"Health monitoring started (interval: {self._check_interval}s, "
            f"timeout: {self._heartbeat_timeout}s)"
        )

    def stop_monitoring(self) -> None:
        """
        Stop background health monitoring thread.
        """
        if not self._monitor_running:
            return

        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logging.info("Health monitoring stopped")

    def _monitor_loop(self) -> None:
        """
        Background monitoring loop that checks health and logs issues.

        Only logs when there are problems to avoid unnecessary log entries.
        """
        while self._monitor_running:
            try:
                self._check_and_log_health()
            except Exception as e:
                logging.error(f"Health monitor error: {e}")

            time.sleep(self._check_interval)

    def _check_and_log_health(self) -> None:
        """
        Check health status, log issues, and attempt recovery if enabled.

        Logs warnings for degraded providers and errors for unhealthy ones.
        Attempts auto-recovery for providers with registered recovery callbacks.
        """
        summary = self.get_summary()
        current_time = time.time()

        # Only log if there are issues
        if summary["overall_status"] == "healthy":
            return

        # Get details about problematic providers
        unhealthy_details = []
        degraded_details = []
        providers_to_recover = []

        with self._lock:
            for name, provider in self._providers.items():
                seconds_since_heartbeat = (
                    round(current_time - provider.last_heartbeat, 1)
                    if provider.last_heartbeat
                    else None
                )

                is_timed_out = (
                    provider.last_heartbeat
                    and current_time - provider.last_heartbeat > self._heartbeat_timeout
                )

                # Check for timeout (unhealthy)
                if is_timed_out:
                    unhealthy_details.append(
                        f"{name} (no heartbeat for {seconds_since_heartbeat}s)"
                    )
                    if self._auto_recovery and name in self._recovery_callbacks:
                        providers_to_recover.append(name)
                elif provider.status == HealthStatus.UNHEALTHY:
                    detail = f"{name}"
                    if provider.last_error:
                        detail += f": {provider.last_error}"
                    unhealthy_details.append(detail)
                    if self._auto_recovery and name in self._recovery_callbacks:
                        providers_to_recover.append(name)
                elif provider.status == HealthStatus.DEGRADED:
                    degraded_details.append(f"{name} ({provider.error_count} errors)")

        # Log unhealthy providers as ERROR
        if unhealthy_details:
            logging.error(f"Unhealthy providers: {', '.join(unhealthy_details)}")

        # Log degraded providers as WARNING
        if degraded_details:
            logging.warning(f"Degraded providers: {', '.join(degraded_details)}")

        # Attempt recovery for unhealthy providers
        for name in providers_to_recover:
            self._attempt_recovery(name)

    def _attempt_recovery(self, name: str) -> bool:
        """
        Attempt to recover an unhealthy provider.

        Parameters
        ----------
        name : str
            Name of the provider to recover.

        Returns
        -------
        bool
            True if recovery was attempted, False if skipped.
        """
        current_time = time.time()

        with self._lock:
            if name not in self._recovery_callbacks:
                return False

            if name not in self._recovery_states:
                self._recovery_states[name] = RecoveryState(
                    max_attempts=self._max_recovery_attempts
                )

            state = self._recovery_states[name]

            # Check if max attempts exceeded
            if state.attempts >= state.max_attempts:
                logging.warning(
                    f"Provider '{name}' exceeded max recovery attempts "
                    f"({state.max_attempts}), giving up"
                )
                return False

            # Check cooldown
            if current_time - state.last_attempt < self._recovery_cooldown:
                remaining = round(
                    self._recovery_cooldown - (current_time - state.last_attempt), 1
                )
                logging.debug(
                    f"Provider '{name}' recovery on cooldown, {remaining}s remaining"
                )
                return False

            # Update recovery state
            state.attempts += 1
            state.last_attempt = current_time

            # Mark provider as recovering
            if name in self._providers:
                self._providers[name].status = HealthStatus.RECOVERING

            callback = self._recovery_callbacks[name]

        # Execute recovery outside lock to prevent deadlocks
        logging.info(
            f"Attempting recovery for '{name}' (attempt {state.attempts}/{state.max_attempts})"
        )

        try:
            success = callback()
            if success:
                logging.info(f"Recovery successful for '{name}'")
                with self._lock:
                    if name in self._recovery_states:
                        self._recovery_states[name].attempts = 0
                return True
            else:
                logging.warning(f"Recovery failed for '{name}'")
                return False
        except Exception as e:
            logging.error(f"Recovery error for '{name}': {e}")
            return False

    def get_recovery_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get recovery status for a provider.

        Parameters
        ----------
        name : str
            Name of the provider.

        Returns
        -------
        dict or None
            Recovery status including attempts and last attempt time,
            or None if provider has no recovery callback.
        """
        with self._lock:
            if name not in self._recovery_states:
                return None

            state = self._recovery_states[name]
            return {
                "attempts": state.attempts,
                "max_attempts": state.max_attempts,
                "last_attempt": state.last_attempt,
                "has_callback": name in self._recovery_callbacks,
            }
