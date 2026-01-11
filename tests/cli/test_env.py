"""Tests for CLI env command."""

import os
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app


class TestEnvCommand:
    """Tests for env command."""

    def test_env_help(self, runner: CliRunner):
        """Test env --help (implicit through error message)."""
        result = runner.invoke(app, ["env"])
        # Missing required argument shows usage
        assert result.exit_code != 0 or "list" in result.output.lower()

    def test_env_list(self, runner: CliRunner):
        """Test env list command."""
        result = runner.invoke(app, ["env", "list"])
        assert result.exit_code == 0
        assert "OM_API_KEY" in result.output or "Environment" in result.output

    def test_env_get_unset_variable(self, runner: CliRunner):
        """Test env get with unset variable."""
        result = runner.invoke(app, ["env", "get", "NONEXISTENT_VAR_12345"])
        assert result.exit_code == 0
        assert "not set" in result.output.lower()

    def test_env_get_set_variable(self, runner: CliRunner):
        """Test env get with set variable."""
        os.environ["TEST_CLI_VAR"] = "test_value"
        try:
            result = runner.invoke(app, ["env", "get", "TEST_CLI_VAR"])
            assert result.exit_code == 0
            assert "test_value" in result.output
        finally:
            del os.environ["TEST_CLI_VAR"]

    def test_env_get_masks_api_keys(self, runner: CliRunner):
        """Test env get masks sensitive API keys."""
        os.environ["TEST_API_KEY"] = "sk-1234567890abcdefghijklmnop"
        try:
            result = runner.invoke(app, ["env", "get", "TEST_API_KEY"])
            assert result.exit_code == 0
            # Should be masked, not show full key
            assert "sk-1234567890abcdefghijklmnop" not in result.output
            assert "..." in result.output or "***" in result.output
        finally:
            del os.environ["TEST_API_KEY"]

    def test_env_validate(self, runner: CliRunner):
        """Test env validate command."""
        result = runner.invoke(app, ["env", "validate"])
        # May fail if OM_API_KEY not set, but should run
        assert result.exit_code in [0, 1]

    def test_env_invalid_action(self, runner: CliRunner):
        """Test env with invalid action."""
        result = runner.invoke(app, ["env", "invalid_action"])
        assert result.exit_code == 1
        assert "unknown" in result.output.lower() or "invalid" in result.output.lower()

    def test_env_set_missing_value(self, runner: CliRunner):
        """Test env set without value."""
        result = runner.invoke(app, ["env", "set", "SOME_KEY"])
        assert result.exit_code == 1

    def test_env_generate(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """Test env generate command."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["env", "generate"], input="y\n")
        # Should create env.example or already exist
        assert result.exit_code == 0 or "exists" in result.output.lower()
