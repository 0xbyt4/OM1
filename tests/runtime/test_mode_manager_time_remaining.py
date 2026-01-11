"""
Tests for time_remaining calculation in ModeManager.

These tests verify that time_remaining is never negative, which could
cause confusion or incorrect behavior in downstream code.
"""

import ast
from pathlib import Path


class TestTimeRemainingSourceAnalysis:
    """Static analysis tests for time_remaining calculation."""

    def test_time_remaining_uses_max_zero(self):
        """Test that time_remaining calculation prevents negative values.

        BUG: If mode_duration exceeds timeout_seconds, time_remaining becomes
        negative. This can cause confusion or incorrect behavior.

        FIXED: Should use max(0, timeout_seconds - mode_duration) to clamp
        the value to non-negative.
        """
        source_file = Path("src/runtime/multi_mode/manager.py")
        assert source_file.exists(), "manager.py not found"

        source_code = source_file.read_text()
        tree = ast.parse(source_code)

        # Find the get_mode_info method
        target_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_mode_info":
                target_method = node
                break

        assert target_method is not None, "get_mode_info method not found"

        # Check if max() is used with the subtraction
        source_lines = source_code.split("\n")
        in_get_mode_info = False
        found_time_remaining = False
        uses_max_protection = False

        for i, line in enumerate(source_lines):
            if "def get_mode_info" in line:
                in_get_mode_info = True
            elif in_get_mode_info and line.strip().startswith("def "):
                in_get_mode_info = False

            if in_get_mode_info and '"time_remaining"' in line:
                found_time_remaining = True

            if in_get_mode_info and found_time_remaining:
                # Check next few lines for the calculation
                if "timeout_seconds" in line and "mode_duration" in line:
                    if "max(" in line or "max(0" in line.replace(" ", ""):
                        uses_max_protection = True
                    break

        assert found_time_remaining, "time_remaining not found in get_mode_info"
        assert uses_max_protection, (
            "BUG FOUND: time_remaining calculation does not use max(0, ...) "
            "to prevent negative values. If mode_duration exceeds timeout_seconds, "
            "the result will be negative."
        )


class TestTimeRemainingBehavior:
    """Tests demonstrating the negative time_remaining issue."""

    def test_subtraction_without_max_can_be_negative(self):
        """Prove that simple subtraction can produce negative values."""
        timeout_seconds = 30
        mode_duration = 45  # Exceeded timeout

        # Without max(), this is negative
        time_remaining_buggy = timeout_seconds - mode_duration
        assert time_remaining_buggy < 0, "Without protection, time can be negative"

    def test_max_zero_prevents_negative(self):
        """Prove that max(0, ...) prevents negative values."""
        timeout_seconds = 30
        mode_duration = 45  # Exceeded timeout

        # With max(), this is clamped to 0
        time_remaining_fixed = max(0, timeout_seconds - mode_duration)
        assert time_remaining_fixed == 0, "max(0, ...) should clamp to 0"
        assert time_remaining_fixed >= 0, "time_remaining should never be negative"

    def test_max_zero_preserves_positive_values(self):
        """Prove that max(0, ...) preserves positive values."""
        timeout_seconds = 30
        mode_duration = 10  # Still within timeout

        time_remaining = max(0, timeout_seconds - mode_duration)
        assert time_remaining == 20, "Positive values should be preserved"
