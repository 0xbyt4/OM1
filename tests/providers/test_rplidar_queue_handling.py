"""
Tests for RPLidar queue handling edge cases.

These tests prove that the queue exception handling pattern needs to catch
both Empty and Full exceptions to handle concurrent access scenarios.
"""

import ast
from pathlib import Path
from queue import Empty, Full, Queue

import pytest


class TestRPLidarSourceCodeAnalysis:
    """Static analysis tests that check the actual source code."""

    def test_rplidar_queue_retry_catches_full_exception(self):
        """Test that rplidar_processor catches Full exception in retry logic.

        This test parses the actual source code to verify that the exception
        handler catches both Empty and Full exceptions.

        BUG: If only Empty is caught, Full exceptions from put_nowait will
        propagate up and cause unnecessary error handling.
        """
        # Read the actual source file
        source_file = Path("src/providers/rplidar_provider.py")
        assert source_file.exists(), "rplidar_provider.py not found"

        source_code = source_file.read_text()

        # Parse the source code
        tree = ast.parse(source_code)

        # Find the rplidar_processor function
        rplidar_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "rplidar_processor":
                rplidar_func = node
                break

        assert rplidar_func is not None, "rplidar_processor function not found"

        # Find exception handlers that catch only Empty after put_nowait
        # We look for the pattern: except Empty: (without Full)
        buggy_handlers = []

        for node in ast.walk(rplidar_func):
            if isinstance(node, ast.ExceptHandler):
                # Check if this handler only catches Empty
                if isinstance(node.type, ast.Name) and node.type.id == "Empty":
                    # Get the line number for reporting
                    buggy_handlers.append(node.lineno)

        # The bug is at line 114 - only catches Empty, should catch (Empty, Full)
        # After fix, there should be no handler that catches only Empty after put_nowait
        #
        # Note: Line 105 catches Empty for get_nowait which is correct
        # But line 114 catches Empty for the retry block which should catch Full too

        # Check if line 114 is in the buggy handlers
        # Line 114 is the problematic one (inside the retry try block after put_nowait)
        has_bug_at_line_114 = 114 in buggy_handlers

        assert not has_bug_at_line_114, (
            f"BUG FOUND: Line 114 only catches Empty exception. "
            f"It should catch (Empty, Full) to handle concurrent queue access. "
            f"Found except handlers at lines: {buggy_handlers}"
        )


class TestQueueExceptionBehavior:
    """Tests proving queue exception behavior."""

    def test_put_nowait_raises_full_on_full_queue(self):
        """Prove that put_nowait raises Full when queue is full."""
        queue = Queue(maxsize=1)
        queue.put("item")

        with pytest.raises(Full):
            queue.put_nowait("another_item")

    def test_get_nowait_raises_empty_on_empty_queue(self):
        """Prove that get_nowait raises Empty when queue is empty."""
        queue = Queue(maxsize=1)

        with pytest.raises(Empty):
            queue.get_nowait()

    def test_concurrent_fill_scenario(self):
        """Prove the race condition scenario that causes the bug."""
        queue = Queue(maxsize=1)
        queue.put("original_item")

        # Step 1: Queue full, put_nowait raises Full
        with pytest.raises(Full):
            queue.put_nowait("new_item")

        # Step 2: Remove item
        queue.get_nowait()

        # Step 3: Concurrent fill
        queue.put("concurrent_item")

        # Step 4: put_nowait raises Full again
        with pytest.raises(Full):
            queue.put_nowait("new_item")
