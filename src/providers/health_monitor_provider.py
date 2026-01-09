import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .singleton import singleton


class HealthStatus(Enum):
    """
    Enumeration for provider health status.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


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
        """
        self._lock: threading.Lock = threading.Lock()
        self._providers: Dict[str, ProviderHealth] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._error_threshold = error_threshold
        self._start_time = time.time()

        logging.info("HealthMonitorProvider initialized")

    def register(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a provider for health monitoring.

        Parameters
        ----------
        name : str
            Unique name for the provider.
        metadata : dict, optional
            Additional metadata about the provider.
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
