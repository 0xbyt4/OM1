from dataclasses import dataclass
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from llm.function_schemas import (
    convert_function_calls_to_actions,
    generate_function_schema_from_action,
)


@dataclass
class SampleInput:
    value: str


@dataclass
class SampleOutput:
    result: str


@dataclass
class SampleInterface(Interface[SampleInput, SampleOutput]):
    input: SampleInput
    output: SampleOutput


class SampleConnector(ActionConnector[ActionConfig, SampleOutput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.last_output: Optional[SampleOutput] = None

    async def connect(self, output_interface: SampleOutput) -> None:
        self.last_output = output_interface


@pytest.fixture
def action_config():
    return ActionConfig()


@pytest.fixture
def test_connector(action_config):
    return SampleConnector(action_config)


@pytest.fixture
def agent_action(test_connector):
    return AgentAction(
        name="test_action",
        llm_label="test_llm_label",
        interface=SampleInterface,
        connector=test_connector,
        exclude_from_prompt=True,
    )


def test_generate_function_schema_from_action(agent_action):
    schema = generate_function_schema_from_action(agent_action)

    assert "function" in schema
    assert schema["type"] == "function"

    fn = schema["function"]
    assert "description" in fn
    assert "parameters" in fn
    assert fn["name"] == "test_llm_label"

    params = fn["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "value" in params["properties"]

    value_prop = params["properties"]["value"]
    assert value_prop["type"] == "string"
    assert "description" in value_prop

    assert params["required"] == ["value"]
    assert fn["description"].startswith("SampleInterface(")


class TestConvertFunctionCallsToActions:
    """Tests for convert_function_calls_to_actions function."""

    def test_single_parameter_action(self):
        """Test conversion with single action parameter."""
        function_calls = [
            {
                "function": {
                    "name": "speak",
                    "arguments": '{"action": "hello world"}',
                }
            }
        ]

        actions = convert_function_calls_to_actions(function_calls)

        assert len(actions) == 1
        assert actions[0].type == "speak"
        assert actions[0].value == "hello world"
        assert actions[0].args == {"action": "hello world"}

    def test_multi_parameter_action(self):
        """Test conversion preserves all function call parameters in args field."""
        function_calls = [
            {
                "function": {
                    "name": "wallet",
                    "arguments": '{"action": "send", "to_address": "0x123abc", "amount": "0.5", "chain": "ethereum"}',
                }
            }
        ]

        actions = convert_function_calls_to_actions(function_calls)

        assert len(actions) == 1
        assert actions[0].type == "wallet"
        assert actions[0].value == "send"
        assert actions[0].args is not None
        assert actions[0].args["action"] == "send"
        assert actions[0].args["to_address"] == "0x123abc"
        assert actions[0].args["amount"] == "0.5"
        assert actions[0].args["chain"] == "ethereum"

    def test_multiple_function_calls(self):
        """Test conversion of multiple function calls."""
        function_calls = [
            {
                "function": {
                    "name": "move",
                    "arguments": '{"action": "forward"}',
                }
            },
            {
                "function": {
                    "name": "speak",
                    "arguments": '{"action": "hello"}',
                }
            },
        ]

        actions = convert_function_calls_to_actions(function_calls)

        assert len(actions) == 2
        assert actions[0].type == "move"
        assert actions[1].type == "speak"

    def test_fallback_to_text_parameter(self):
        """Test fallback to text parameter when action is missing."""
        function_calls = [
            {
                "function": {
                    "name": "speak",
                    "arguments": '{"text": "hello world"}',
                }
            }
        ]

        actions = convert_function_calls_to_actions(function_calls)

        assert len(actions) == 1
        assert actions[0].value == "hello world"
        assert actions[0].args["text"] == "hello world"

    def test_args_dict_as_object(self):
        """Test that args field is passed as dict object, not string."""
        function_calls = [
            {
                "function": {
                    "name": "test",
                    "arguments": {"action": "test_value", "param1": "value1"},
                }
            }
        ]

        actions = convert_function_calls_to_actions(function_calls)

        assert len(actions) == 1
        assert isinstance(actions[0].args, dict)
        assert actions[0].args["param1"] == "value1"

    def test_empty_function_calls(self):
        """Test conversion with empty list."""
        actions = convert_function_calls_to_actions([])
        assert len(actions) == 0

    def test_invalid_json_skipped(self):
        """Test that invalid JSON arguments are skipped."""
        function_calls = [
            {
                "function": {
                    "name": "test",
                    "arguments": "not valid json",
                }
            }
        ]

        actions = convert_function_calls_to_actions(function_calls)
        assert len(actions) == 0
