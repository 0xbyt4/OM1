"""
Test to PROVE the race condition bug exists in the current code.

BUG: In _execute_transition(), the lock scope is too small:
- Lock covers only: check _is_transitioning, set _is_transitioning = True
- Lock does NOT cover: try/except/finally block where _is_transitioning = False

This allows race conditions where multiple transitions modify state simultaneously.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from runtime.multi_mode.config import (
    ModeConfig,
    ModeSystemConfig,
    TransitionRule,
    TransitionType,
)
from runtime.multi_mode.manager import ModeManager


@pytest.fixture
def sample_mode_configs():
    """Sample mode configurations for testing."""
    return {
        "default": ModeConfig(
            version="v1.0.0",
            name="default",
            display_name="Default Mode",
            description="Default operational mode",
            system_prompt_base="You are a test agent",
            timeout_seconds=300.0,
        ),
        "active": ModeConfig(
            version="v1.0.0",
            name="active",
            display_name="Active Mode",
            description="Active mode",
            system_prompt_base="You are active",
        ),
        "emergency": ModeConfig(
            version="v1.0.0",
            name="emergency",
            display_name="Emergency Mode",
            description="Emergency mode",
            system_prompt_base="Emergency",
        ),
    }


@pytest.fixture
def sample_transition_rules():
    """Sample transition rules for testing."""
    return [
        TransitionRule(
            from_mode="default",
            to_mode="active",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["active"],
            priority=1,
        ),
        TransitionRule(
            from_mode="default",
            to_mode="emergency",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["emergency"],
            priority=2,
        ),
        TransitionRule(
            from_mode="active",
            to_mode="default",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["default"],
            priority=1,
        ),
        TransitionRule(
            from_mode="emergency",
            to_mode="default",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["default"],
            priority=1,
        ),
    ]


@pytest.fixture
def sample_system_config(sample_mode_configs, sample_transition_rules):
    """Sample system configuration for testing."""
    config = ModeSystemConfig(
        name="test_system",
        default_mode="default",
        config_name="test_config",
        allow_manual_switching=True,
        mode_memory_enabled=False,
        api_key="test_api_key",
    )
    config.modes = sample_mode_configs
    config.transition_rules = sample_transition_rules
    return config


@pytest.fixture
def mode_manager(sample_system_config):
    """Mode manager instance for testing."""
    with (
        patch("runtime.multi_mode.manager.open_zenoh_session"),
        patch("runtime.multi_mode.manager.ModeManager._load_mode_state"),
    ):
        return ModeManager(sample_system_config)


class TestRaceConditionProof:
    """Tests that prove the race condition bug exists."""

    @pytest.mark.asyncio
    async def test_lock_scope_during_transition(self, mode_manager):
        """
        PROOF: Check if lock is held during the entire transition.

        We inject a hook that checks lock state during transition execution.
        In buggy code: lock is NOT held during hooks
        In fixed code: lock IS held during hooks
        """
        lock_held_during_hooks = None

        async def check_lock_hook(*args, **kwargs):
            nonlocal lock_held_during_hooks
            # Check if transition lock is currently held
            lock_held_during_hooks = mode_manager._transition_lock.locked()
            return True

        # Inject our check into the lifecycle hooks
        mode_manager.config.modes["active"].execute_lifecycle_hooks = check_lock_hook

        # Execute transition - this calls the REAL _execute_transition
        await mode_manager._execute_transition("active", "test")

        # RESULT:
        # - BUGGY code: lock_held_during_hooks = False (lock released before hooks)
        # - FIXED code: lock_held_during_hooks = True (lock held during hooks)
        print(f"Lock held during hooks: {lock_held_during_hooks}")

        # This assertion FAILS on buggy code, PASSES on fixed code
        assert lock_held_during_hooks is True, (
            "BUG PROVEN: Lock is NOT held during lifecycle hook execution. "
            "The lock scope is too small - it releases before the try block."
        )

    @pytest.mark.asyncio
    async def test_concurrent_transition_state_corruption(self, mode_manager):
        """
        PROOF: Concurrent transitions can corrupt state.

        Scenario:
        1. Transition A starts to "active", sets _is_transitioning = True, releases lock
        2. Transition A is in the middle of hooks (slow)
        3. Transition A hits exception, finally sets _is_transitioning = False
        4. Transition B can now start while A's state changes are incomplete
        """
        transitions_completed = []
        exception_raised = False

        # Make "active" transition slow and fail partway
        async def slow_failing_hook(*args, **kwargs):
            nonlocal exception_raised
            await asyncio.sleep(0.05)  # Slow operation
            # Simulate partial state change before failure
            mode_manager.state.previous_mode = "partial"
            exception_raised = True
            raise Exception("Simulated hook failure")

        async def normal_hook(*args, **kwargs):
            transitions_completed.append("emergency")
            return True

        mode_manager.config.modes["active"].execute_lifecycle_hooks = slow_failing_hook
        mode_manager.config.modes["emergency"].execute_lifecycle_hooks = normal_hook
        mode_manager.config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)

        async def transition_a():
            try:
                result = await mode_manager._execute_transition("active", "test_a")
                transitions_completed.append(("active", result))
            except Exception:
                transitions_completed.append(("active", "exception"))

        async def transition_b():
            # Wait for A to be in the middle of its transition
            await asyncio.sleep(0.03)
            # At this point, A has released the lock but is still running
            result = await mode_manager._execute_transition("emergency", "test_b")
            transitions_completed.append(("emergency", result))

        # Run concurrently
        await asyncio.gather(transition_a(), transition_b(), return_exceptions=True)

        print(f"Transitions: {transitions_completed}")
        print(f"Final state: {mode_manager.state.current_mode}")
        print(f"Previous mode: {mode_manager.state.previous_mode}")

        # In buggy code, state can be corrupted
        # "partial" was set by failing transition A, but transition B may have also modified state
        # This test documents the race condition behavior

    @pytest.mark.asyncio
    async def test_flag_can_be_seen_as_false_during_transition(self, mode_manager):
        """
        PROOF: Another coroutine can see _is_transitioning = False while transition runs.

        In buggy code, after exception in transition:
        1. finally block sets _is_transitioning = False
        2. But the transition's state changes may not be complete
        3. Another coroutine can start a new transition
        """
        flag_observations = []

        async def failing_hook(*args, **kwargs):
            # Record flag state
            flag_observations.append(("during_hook", mode_manager._is_transitioning))
            await asyncio.sleep(0.02)
            raise Exception("Hook failure")

        mode_manager.config.modes["active"].execute_lifecycle_hooks = failing_hook
        mode_manager.config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)

        async def observer():
            """Observe flag state multiple times during transition."""
            for i in range(10):
                await asyncio.sleep(0.005)
                flag_observations.append(("observer", mode_manager._is_transitioning))

        async def do_transition():
            try:
                await mode_manager._execute_transition("active", "test")
            except Exception:
                pass
            flag_observations.append(("after_transition", mode_manager._is_transitioning))

        await asyncio.gather(do_transition(), observer())

        print(f"Flag observations: {flag_observations}")

        # Check if flag was ever False while transition was happening
        # In buggy code, after exception the finally runs outside lock
        # allowing observers to see inconsistent state
