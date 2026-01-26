from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from llm.function_schemas import (
    convert_function_calls_to_actions,
    generate_function_schema_from_action,
    generate_function_schemas_from_actions,
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


class SampleStrEnum(str, Enum):
    WALK = "walk"
    RUN = "run"
    STOP = "stop"


class SampleIntEnum(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class StrEnumFieldInput:
    action: SampleStrEnum


@dataclass
class StrEnumFieldInterface(Interface[StrEnumFieldInput, StrEnumFieldInput]):
    """Action with a string enum parameter."""

    input: StrEnumFieldInput
    output: StrEnumFieldInput


@dataclass
class IntEnumFieldInput:
    priority: SampleIntEnum


@dataclass
class IntEnumFieldInterface(Interface[IntEnumFieldInput, IntEnumFieldInput]):
    """Action with an integer enum parameter."""

    input: IntEnumFieldInput
    output: IntEnumFieldInput


@dataclass
class MixedFieldInput:
    action: str
    count: int
    speed: float
    enabled: bool


@dataclass
class MixedFieldInterface(Interface[MixedFieldInput, MixedFieldInput]):
    """Action with mixed parameter types."""

    input: MixedFieldInput
    output: MixedFieldInput


@pytest.fixture
def action_config():
    return ActionConfig()


@pytest.fixture
def test_connector(action_config):
    return SampleConnector(action_config)


def _make_agent_action(test_connector, interface, label="test_label"):
    return AgentAction(
        name="test_action",
        llm_label=label,
        interface=interface,
        connector=test_connector,
        exclude_from_prompt=False,
    )


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


def test_str_enum_field_generates_enum(test_connector):
    action = _make_agent_action(
        test_connector, StrEnumFieldInterface, "str_enum_action"
    )
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert props["action"]["type"] == "string"
    assert "enum" in props["action"]
    assert set(props["action"]["enum"]) == {"walk", "run", "stop"}


def test_int_enum_field_does_not_crash(test_connector):
    action = _make_agent_action(
        test_connector, IntEnumFieldInterface, "int_enum_action"
    )
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert "priority" in props
    assert "enum" in props["priority"]
    assert set(props["priority"]["enum"]) == {1, 2, 3}


def test_mixed_field_types(test_connector):
    action = _make_agent_action(test_connector, MixedFieldInterface, "mixed_action")
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert props["action"]["type"] == "string"
    assert (
        props["count"]["type"] == "integer"
    ), f"int field should produce 'integer', got '{props['count']['type']}'"
    assert (
        props["speed"]["type"] == "number"
    ), f"float field should produce 'number', got '{props['speed']['type']}'"
    assert (
        props["enabled"]["type"] == "boolean"
    ), f"bool field should produce 'boolean', got '{props['enabled']['type']}'"

    required = schema["function"]["parameters"]["required"]
    assert set(required) == {"action", "count", "speed", "enabled"}


def test_schemas_excludes_excluded_actions(test_connector):
    action_included = _make_agent_action(
        test_connector, SampleInterface, "included_action"
    )
    action_excluded = AgentAction(
        name="excluded",
        llm_label="excluded_action",
        interface=SampleInterface,
        connector=test_connector,
        exclude_from_prompt=True,
    )
    schemas = generate_function_schemas_from_actions([action_included, action_excluded])
    names = [s["function"]["name"] for s in schemas]
    assert "included_action" in names
    assert "excluded_action" not in names


def test_convert_function_calls_basic():
    calls = [
        {
            "function": {
                "name": "move",
                "arguments": '{"action": "walk"}',
            }
        }
    ]
    actions = convert_function_calls_to_actions(calls)
    assert len(actions) == 1
    assert actions[0].type == "move"
    assert actions[0].value == "walk"


def test_convert_function_calls_invalid_json():
    calls = [
        {
            "function": {
                "name": "move",
                "arguments": "not valid json{",
            }
        }
    ]
    actions = convert_function_calls_to_actions(calls)
    assert len(actions) == 0


def test_convert_function_calls_fallback_params():
    calls = [
        {
            "function": {
                "name": "speak",
                "arguments": '{"text": "hello world"}',
            }
        }
    ]
    actions = convert_function_calls_to_actions(calls)
    assert len(actions) == 1
    assert actions[0].value == "hello world"
