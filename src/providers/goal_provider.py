import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .singleton import singleton


@dataclass
class Goal:
    """
    Represents a behavioral goal for the robot.

    Parameters
    ----------
    name : str
        The goal name/action (e.g., "guard the house").
    priority : str
        Priority level: "low", "medium", "high", or "critical".
    description : str
        Additional context about the goal.
    created_at : str
        ISO timestamp when the goal was created.
    status : str
        Goal status: "active", "completed", "cancelled".
    """

    name: str
    priority: str = "medium"
    description: str = ""
    created_at: str = ""
    status: str = "active"

    def __post_init__(self):
        """Initialize created_at timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@singleton
class GoalProvider:
    """
    Provider that manages behavioral goals for the robot.

    Goals can be persisted via an HTTP API (if configured) or stored locally in memory.
    This provider follows the same pattern as UnitreeGo2LocationsProvider.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 5,
        refresh_interval: int = 30,
    ):
        """
        Initialize the goal provider.

        Parameters
        ----------
        base_url : str
            The HTTP endpoint for goals API. If empty, goals are stored locally only.
        timeout : int
            Timeout for HTTP requests in seconds.
        refresh_interval : int
            How often to refresh goals from API in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.refresh_interval = refresh_interval
        self._goals: Dict[str, Goal] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._current_goal: Optional[str] = None

    def start(self) -> None:
        """
        Start the background fetch thread (only if base_url is configured).
        """
        if not self.base_url:
            logging.info("GoalProvider running in local-only mode (no API configured)")
            return

        if self._thread and self._thread.is_alive():
            logging.warning("GoalProvider already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("GoalProvider background thread started")

    def stop(self) -> None:
        """
        Stop the background fetch thread.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """
        Background thread that periodically fetches goals from API.
        """
        while not self._stop_event.is_set():
            try:
                self._fetch()
            except Exception:
                logging.exception("Error fetching goals")

            self._stop_event.wait(timeout=self.refresh_interval)

    def _fetch(self) -> None:
        """
        Fetch goals from the API and update cache.
        """
        if not self.base_url:
            return

        try:
            resp = requests.get(self.base_url, timeout=self.timeout)

            if resp.status_code < 200 or resp.status_code >= 300:
                logging.error(f"Goals API returned {resp.status_code}: {resp.text}")
                return

            data = resp.json()
            self._update_goals(data)

        except requests.RequestException:
            logging.exception("Error fetching goals from API")

    def _update_goals(self, goals_raw: Dict) -> None:
        """
        Parse and store goals from API response.

        Parameters
        ----------
        goals_raw : Dict
            Raw goals data from the API.
        """
        parsed = {}

        if isinstance(goals_raw, dict):
            goals_list = goals_raw.get("goals", goals_raw)
            if isinstance(goals_list, list):
                for item in goals_list:
                    if not isinstance(item, dict):
                        continue
                    name = (item.get("name") or "").strip()
                    if not name:
                        continue
                    parsed[name.lower()] = Goal(
                        name=name,
                        priority=item.get("priority", "medium"),
                        description=item.get("description", ""),
                        created_at=item.get("created_at", ""),
                        status=item.get("status", "active"),
                    )
            elif isinstance(goals_list, dict):
                for k, v in goals_list.items():
                    if isinstance(v, dict):
                        parsed[k.strip().lower()] = Goal(
                            name=v.get("name", k),
                            priority=v.get("priority", "medium"),
                            description=v.get("description", ""),
                            created_at=v.get("created_at", ""),
                            status=v.get("status", "active"),
                        )

        with self._lock:
            self._goals.update(parsed)

    def set_goal(
        self,
        name: str,
        priority: str = "medium",
        description: str = "",
    ) -> Goal:
        """
        Set a new goal or update an existing one.

        Parameters
        ----------
        name : str
            The goal name/action.
        priority : str
            Priority level: "low", "medium", "high", or "critical".
        description : str
            Additional context about the goal.

        Returns
        -------
        Goal
            The created or updated goal.
        """
        key = name.strip().lower()
        goal = Goal(name=name, priority=priority, description=description)

        with self._lock:
            self._goals[key] = goal
            if priority in ("high", "critical") or self._current_goal is None:
                self._current_goal = key

        logging.info(f"Goal set: {name} (priority: {priority})")

        if self.base_url:
            self._persist_goal(goal)

        return goal

    def _persist_goal(self, goal: Goal) -> None:
        """
        Persist a goal to the API.

        Parameters
        ----------
        goal : Goal
            The goal to persist.
        """
        if not self.base_url:
            return

        try:
            resp = requests.post(
                self.base_url,
                json=asdict(goal),
                timeout=self.timeout,
            )
            if resp.status_code >= 300:
                logging.error(f"Failed to persist goal: {resp.status_code}")
        except requests.RequestException:
            logging.exception("Error persisting goal to API")

    def get_goal(self, name: str) -> Optional[Goal]:
        """
        Get a specific goal by name.

        Parameters
        ----------
        name : str
            The goal name to retrieve.

        Returns
        -------
        Goal or None
            The goal if found, otherwise None.
        """
        if not name:
            return None
        key = name.strip().lower()
        with self._lock:
            return self._goals.get(key)

    def get_all_goals(self) -> Dict[str, Goal]:
        """
        Get all cached goals.

        Returns
        -------
        Dict[str, Goal]
            A dictionary of all goals keyed by their names.
        """
        with self._lock:
            return dict(self._goals)

    def get_active_goals(self) -> List[Goal]:
        """
        Get all active goals sorted by priority.

        Returns
        -------
        List[Goal]
            List of active goals sorted by priority (critical > high > medium > low).
        """
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        with self._lock:
            active = [g for g in self._goals.values() if g.status == "active"]
            return sorted(active, key=lambda g: priority_order.get(g.priority, 2))

    def get_current_goal(self) -> Optional[Goal]:
        """
        Get the current highest-priority active goal.

        Returns
        -------
        Goal or None
            The current goal if any active goals exist.
        """
        with self._lock:
            if self._current_goal and self._current_goal in self._goals:
                return self._goals[self._current_goal]
            active = self.get_active_goals()
            return active[0] if active else None

    def complete_goal(self, name: str) -> bool:
        """
        Mark a goal as completed.

        Parameters
        ----------
        name : str
            The goal name to complete.

        Returns
        -------
        bool
            True if goal was found and completed, False otherwise.
        """
        key = name.strip().lower()
        with self._lock:
            if key in self._goals:
                self._goals[key].status = "completed"
                if self._current_goal == key:
                    self._current_goal = None
                logging.info(f"Goal completed: {name}")
                return True
        return False

    def cancel_goal(self, name: str) -> bool:
        """
        Mark a goal as cancelled.

        Parameters
        ----------
        name : str
            The goal name to cancel.

        Returns
        -------
        bool
            True if goal was found and cancelled, False otherwise.
        """
        key = name.strip().lower()
        with self._lock:
            if key in self._goals:
                self._goals[key].status = "cancelled"
                if self._current_goal == key:
                    self._current_goal = None
                logging.info(f"Goal cancelled: {name}")
                return True
        return False

    def clear_all_goals(self) -> None:
        """
        Clear all goals.
        """
        with self._lock:
            self._goals.clear()
            self._current_goal = None
        logging.info("All goals cleared")

    def to_prompt_context(self) -> str:
        """
        Generate a prompt context string describing current goals.

        Returns
        -------
        str
            A formatted string describing active goals for LLM context.
        """
        active = self.get_active_goals()
        if not active:
            return "You have no active goals."

        lines = ["Your current goals (in priority order):"]
        for i, goal in enumerate(active, 1):
            line = f"{i}. [{goal.priority.upper()}] {goal.name}"
            if goal.description:
                line += f" - {goal.description}"
            lines.append(line)

        return "\n".join(lines)
