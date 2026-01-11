"""Tests for CLI setup command."""

from typer.testing import CliRunner

from cli.main import app


class TestSetupCommand:
    """Tests for setup command."""

    def test_setup_help(self, runner: CliRunner):
        """Test setup --help."""
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--minimal" in result.output
        assert "--full" in result.output
        assert "--config" in result.output

    def test_setup_options_documented(self, runner: CliRunner):
        """Test that setup options are documented."""
        result = runner.invoke(app, ["setup", "--help"])
        assert "--skip-deps" in result.output
        assert "wizard" in result.output.lower() or "setup" in result.output.lower()
