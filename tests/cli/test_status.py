"""Tests for CLI status command."""

from typer.testing import CliRunner

from cli.main import app


class TestStatusCommand:
    """Tests for status command."""

    def test_status_help(self, runner: CliRunner):
        """Test status --help."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--watch" in result.output

    def test_status_no_agents(self, runner: CliRunner):
        """Test status when no agents are running."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show "no agents" or a table
        assert (
            "no" in result.output.lower()
            or "running" in result.output.lower()
            or "agent" in result.output.lower()
        )

    def test_status_provides_helpful_info(self, runner: CliRunner):
        """Test that status provides helpful commands."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should mention how to start an agent or show table
        assert "run" in result.output.lower() or "config" in result.output.lower()
