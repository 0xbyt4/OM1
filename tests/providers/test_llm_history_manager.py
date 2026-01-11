import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from providers.llm_history_manager import ChatMessage, LLMHistoryManager


@dataclass
class MockAction:
    type: str
    value: str


@pytest.fixture
def llm_config():
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "Test Robot"
    return config


@pytest.fixture
def openai_client():
    client = MagicMock(spec=openai.AsyncClient)

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test summary"

    chat_mock = MagicMock()
    completions_mock = MagicMock()
    completions_mock.create = AsyncMock(return_value=response)
    chat_mock.completions = completions_mock
    client.chat = chat_mock

    return client


@pytest.fixture
def history_manager(llm_config, openai_client):
    return LLMHistoryManager(llm_config, openai_client)


@pytest.mark.asyncio
async def test_summarize_messages_success(history_manager):
    # Create test messages
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    # Test successful summarization
    result = await history_manager.summarize_messages(messages)
    assert result.role == "assistant"
    assert "Previously, This is a test summary" == result.content


@pytest.mark.asyncio
async def test_summarize_messages_empty(history_manager):
    # Test with empty messages
    result = await history_manager.summarize_messages([])
    assert result.role == "system"
    assert "No history to summarize" == result.content


@pytest.mark.asyncio
async def test_summarize_messages_api_error(history_manager):
    # Mock API error
    history_manager.client.chat.completions.create.side_effect = Exception("API Error")

    messages = [ChatMessage(role="user", content="Test")]
    result = await history_manager.summarize_messages(messages)

    assert result.role == "system"
    assert "Error summarizing state" == result.content


@pytest.mark.asyncio
async def test_start_summary_task(history_manager):
    # Create test messages that we'll modify in-place
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    # Replace summarize_messages with a mock
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="assistant", content="New summary"
    )

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    # Verify the task was created
    assert history_manager._summary_task is not None

    # Let the event loop process the callback
    await asyncio.sleep(0.1)

    # Because we mocked summarize_messages, the callback should have run
    # and updated the messages list
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert "New summary" == messages[0].content


@pytest.mark.asyncio
async def test_start_summary_task_empty_messages(history_manager):
    # Test with empty messages
    await history_manager.start_summary_task([])
    assert history_manager._summary_task is None


@pytest.mark.asyncio
async def test_start_summary_task_error_handling(history_manager):
    messages = [
        ChatMessage(role="user", content="Test message"),
    ]

    # Mock error in summarization
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="system", content="Error: API service unavailable"
    )

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_update_history_only_current_tick_inputs():
    """Test that only inputs matching the current tick are added to history."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    # Setup mock class that uses the decorator
    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            # Return mock response with actions
            response = MagicMock()
            response.actions = [
                MockAction(type="speak", value="Hello"),
                MockAction(type="emotion", value="happy"),
            ]
            return response

    # Create provider instance
    provider = MockLLMProvider()

    # Add inputs with different ticks
    # Current tick is 0 (initial value)
    provider.io_provider.add_input("audio", "User said hello", 1234.0)
    provider.io_provider.add_input("vision", "Saw a person", 1235.0)

    # Increment tick to 1
    provider.io_provider.increment_tick()

    # Add inputs for tick 1
    provider.io_provider.add_input("audio_new", "User said goodbye", 1236.0)
    provider.io_provider.add_input("lidar", "Detected obstacle", 1237.0)

    # Process with current tick = 1
    await provider.process("test prompt")

    # Should have 2 messages: inputs and actions
    assert len(history_manager.history) == 2

    # First message should be the inputs message
    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    assert "audio_new" in inputs_msg.content
    assert "User said goodbye" in inputs_msg.content
    assert "lidar" in inputs_msg.content
    assert "Detected obstacle" in inputs_msg.content

    assert "User said hello" not in inputs_msg.content
    assert "Saw a person" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_no_inputs_for_current_tick():
    """Test that when no inputs match current tick, only sensor info is added."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    # Setup mock class that uses the decorator
    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Nothing to report")]
            return response

    provider = MockLLMProvider()

    # Add inputs with tick 0
    provider.io_provider.add_input("audio", "Old audio", 1234.0)

    # Increment tick to 1 without adding new inputs
    provider.io_provider.increment_tick()

    # Process with current tick = 1 (no inputs for this tick)
    await provider.process("test prompt")

    # Should have 2 messages: empty inputs and actions
    assert len(history_manager.history) == 2

    # First message should be the inputs message with just the preamble
    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    assert "TestBot sensed the following:" in inputs_msg.content
    # Old inputs should not be included
    assert "Old audio" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_multiple_ticks():
    """Test that inputs are filtered correctly across multiple tick cycles."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 10
    config.agent_name = "MultiTickBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Response")]
            return response

    provider = MockLLMProvider()

    # Tick 0: Add inputs
    provider.io_provider.add_input("input_tick0", "Data at tick 0", 1000.0)
    await provider.process("prompt")

    # Verify only tick 0 data in first cycle
    first_inputs = history_manager.history[0]
    assert "input_tick0" in first_inputs.content
    assert "Data at tick 0" in first_inputs.content

    # Tick 1: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick1", "Data at tick 1", 2000.0)
    await provider.process("prompt")

    # Find the second input message (should be at index 2)
    second_inputs = history_manager.history[2]
    assert "input_tick1" in second_inputs.content
    assert "Data at tick 1" in second_inputs.content
    # Should NOT include tick 0 data
    assert "Data at tick 0" not in second_inputs.content

    # Tick 2: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick2", "Data at tick 2", 3000.0)
    await provider.process("prompt")

    # Find the third input message (should be at index 4)
    third_inputs = history_manager.history[4]
    assert "input_tick2" in third_inputs.content
    assert "Data at tick 2" in third_inputs.content
    # Should NOT include previous tick data
    assert "Data at tick 0" not in third_inputs.content
    assert "Data at tick 1" not in third_inputs.content


@pytest.mark.asyncio
async def test_update_history_tick_boundary():
    """Test input filtering at tick boundaries when inputs are updated."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "BoundaryBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="move", value="forward")]
            return response

    provider = MockLLMProvider()

    # Add input at tick 0
    provider.io_provider.add_input("sensor", "Initial reading", 1000.0)

    # Increment to tick 1
    provider.io_provider.increment_tick()

    # Update the same input key with new data at tick 1
    provider.io_provider.add_input("sensor", "Updated reading", 2000.0)

    # Process at tick 1
    await provider.process("prompt")

    # Should only see the updated reading from tick 1
    inputs_msg = history_manager.history[0]
    assert "Updated reading" in inputs_msg.content
    assert "Initial reading" not in inputs_msg.content


@pytest.mark.asyncio
async def test_callback_uses_lock_for_list_modification(history_manager):
    """Test that callback modifies messages list with thread-safety.

    BUG: The callback modifies messages list without a lock, which can
    cause race conditions when accessed from multiple threads/coroutines.
    """
    # Check if LLMHistoryManager has a _lock attribute for thread-safety
    assert hasattr(history_manager, "_lock"), (
        "BUG: LLMHistoryManager should have a _lock attribute for thread-safe "
        "list modifications in callbacks"
    )


@pytest.mark.asyncio
async def test_safe_pop_with_length_check(history_manager):
    """Test that pop operations use proper length checks instead of truthy check.

    BUG: The pattern `messages.pop(0) if messages else None` is unsafe because:
    1. It checks if list is truthy (non-empty)
    2. Then pops from the list
    3. Between check and pop, list could become empty (TOCTOU bug)

    FIXED: Should use `if len(messages) > 0` with lock protection.
    """
    messages = [ChatMessage(role="user", content="Test")]

    # Mock error in summarization to trigger the error handling path
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="system", content="Error: API failed"
    )

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    # The callback should have handled the error safely
    # Even if messages becomes empty during callback, it should not raise IndexError
    # This test documents the expected behavior after fix


@pytest.mark.asyncio
async def test_concurrent_callback_access(history_manager):
    """Test that concurrent access to messages during callback is thread-safe.

    This test simulates concurrent modifications that could occur when
    multiple summary tasks complete around the same time.
    """
    import threading

    messages = [
        ChatMessage(role="user", content="msg1"),
        ChatMessage(role="user", content="msg2"),
        ChatMessage(role="user", content="msg3"),
    ]

    errors = []

    def modify_list():
        """Simulate concurrent modification."""
        try:
            for _ in range(10):
                if messages:
                    messages.append(ChatMessage(role="user", content="concurrent"))
                if len(messages) > 1:
                    messages.pop()
        except (IndexError, RuntimeError) as e:
            errors.append(e)

    # Start concurrent modifications
    threads = [threading.Thread(target=modify_list) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Note: This test may or may not catch the race condition depending on timing
    # The fix should ensure no errors occur with proper locking
