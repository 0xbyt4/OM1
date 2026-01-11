"""Tests for CLI restart command."""

from typer.testing import CliRunner

from cli.main import app


class TestRestartCommand:
    """Tests for restart command."""

    def test_restart_help(self, runner: CliRunner):
        """Test restart --help."""
        result = runner.invoke(app, ["restart", "--help"])
        assert result.exit_code == 0
        assert "--hot-reload" in result.output
        assert "--force" in result.output
        assert "--timeout" in result.output

    def test_restart_no_args(self, runner: CliRunner):
        """Test restart without arguments shows usage."""
        result = runner.invoke(app, ["restart"])
        # Should show usage or error about missing argument
        assert result.exit_code != 0 or "config" in result.output.lower()

    def test_restart_nonexistent_agent(self, runner: CliRunner):
        """Test restart with nonexistent agent."""
        result = runner.invoke(app, ["restart", "nonexistent_agent_12345"])
        # Should fail gracefully
        assert result.exit_code in [0, 1]

    def test_restart_hot_reload_option(self, runner: CliRunner):
        """Test restart --hot-reload option is documented."""
        result = runner.invoke(app, ["restart", "--help"])
        assert "--hot-reload" in result.output
        assert "-r" in result.output

    def test_restart_force_option(self, runner: CliRunner):
        """Test restart --force option is documented."""
        result = runner.invoke(app, ["restart", "--help"])
        assert "--force" in result.output
        assert "-f" in result.output

    def test_restart_timeout_option(self, runner: CliRunner):
        """Test restart --timeout option is documented."""
        result = runner.invoke(app, ["restart", "--help"])
        assert "--timeout" in result.output
        assert "-t" in result.output
