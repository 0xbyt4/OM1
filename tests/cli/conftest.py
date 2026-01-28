import os
import shutil
import tempfile
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def temp_config_file(temp_dir: str) -> Generator[str, None, None]:
    config_path = os.path.join(temp_dir, "test_config.json5")
    yield config_path


@pytest.fixture
def valid_single_mode_config() -> dict:
    return {
        "version": "v1.0.0",
        "hertz": 1,
        "name": "test_config",
        "api_key": "test_key",
        "system_prompt_base": "Test prompt",
        "system_governance": "Test governance",
        "system_prompt_examples": "Test examples",
        "agent_inputs": [],
        "agent_actions": [],
        "cortex_llm": {
            "type": "OpenAILLM",
            "config": {
                "agent_name": "TestAgent",
            },
        },
    }


@pytest.fixture
def valid_multi_mode_config() -> dict:
    return {
        "version": "v1.0.0",
        "name": "test_multi_mode",
        "default_mode": "idle",
        "allow_manual_switching": True,
        "mode_memory_enabled": False,
        "modes": {
            "idle": {
                "display_name": "Idle Mode",
                "description": "Default idle state",
                "hertz": 1,
                "system_prompt_base": "You are idle.",
                "agent_inputs": [],
                "agent_actions": [],
            },
        },
        "transition_rules": [],
        "cortex_llm": {
            "type": "OpenAILLM",
            "config": {
                "agent_name": "TestAgent",
            },
        },
    }
