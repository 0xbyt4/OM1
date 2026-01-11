"""Tests for CLI modes command."""

from typer.testing import CliRunner

from cli.main import app


class TestModesCommand:
    """Tests for modes command."""

    def test_modes_help(self, runner: CliRunner):
        """Test modes --help."""
        result = runner.invoke(app, ["modes", "--help"])
        assert result.exit_code == 0
        assert "mode-aware" in result.output.lower()
        assert "--json" in result.output

    def test_modes_nonexistent_config(self, runner: CliRunner):
        """Test modes with nonexistent config."""
        result = runner.invoke(app, ["modes", "nonexistent_config_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_modes_non_mode_aware_config(self, runner: CliRunner):
        """Test modes with a non-mode-aware config."""
        result = runner.invoke(app, ["modes", "ollama"])
        assert result.exit_code == 1
        assert "not mode-aware" in result.output

    def test_modes_json_option(self, runner: CliRunner):
        """Test that --json option is documented."""
        result = runner.invoke(app, ["modes", "--help"])
        assert "--json" in result.output
        assert "-j" in result.output
