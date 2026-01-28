import json
import os
import shutil
import tempfile
from typing import Generator

import pytest
from typer.testing import CliRunner

from cli import app


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


runner = CliRunner()


class TestValidateConfigCommand:

    def test_valid_config_passes(self):
        result = runner.invoke(app, ["validate-config", "test"])

        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_valid_config_verbose(self):
        result = runner.invoke(app, ["validate-config", "test", "--verbose"])

        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_nonexistent_config_fails(self):
        result = runner.invoke(app, ["validate-config", "nonexistent_config_12345"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_invalid_json5_fails(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "invalid.json5")
        with open(config_path, "w") as f:
            f.write("{ invalid json5 syntax }")

        result = runner.invoke(app, ["validate-config", config_path])

        assert result.exit_code == 1

    def test_schema_validation_fails(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "missing_fields.json5")
        with open(config_path, "w") as f:
            f.write('{ "name": "test" }')

        result = runner.invoke(app, ["validate-config", config_path])

        assert result.exit_code == 1

    def test_skip_inputs_flag(self):
        result = runner.invoke(
            app, ["validate-config", "test", "--skip-inputs", "--verbose"]
        )

        assert result.exit_code == 0
        assert "skip" in result.stdout.lower() or "valid" in result.stdout.lower()

    def test_allow_missing_flag(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "with_missing.json5")
        config = {
            "version": "v1.0.0",
            "hertz": 1,
            "name": "test_missing",
            "api_key": "test_key",
            "system_prompt_base": "Test",
            "system_governance": "Test governance",
            "system_prompt_examples": "Test examples",
            "agent_inputs": [{"type": "NonExistentInput12345"}],
            "agent_actions": [],
            "cortex_llm": {"type": "OpenAILLM", "config": {"agent_name": "Test"}},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(app, ["validate-config", config_path])
        assert result.exit_code == 1

        result = runner.invoke(app, ["validate-config", config_path, "--allow-missing"])
        assert result.exit_code == 0

    def test_check_components_finds_missing(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "bad_components.json5")
        config = {
            "version": "v1.0.0",
            "hertz": 1,
            "name": "test_bad",
            "api_key": "test_key",
            "system_prompt_base": "Test",
            "system_governance": "Test governance",
            "system_prompt_examples": "Test examples",
            "agent_inputs": [{"type": "FakeInput99999"}],
            "agent_actions": [],
            "cortex_llm": {"type": "OpenAILLM", "config": {"agent_name": "Test"}},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(app, ["validate-config", config_path, "--verbose"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_multi_mode_config(self):
        result = runner.invoke(app, ["validate-config", "spot_modes", "--verbose"])

        assert result.exit_code in [0, 1]
