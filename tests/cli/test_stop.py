"""Tests for CLI stop command."""

from typer.testing import CliRunner

from cli.main import app


class TestStopCommand:
    """Tests for stop command."""

    def test_stop_help(self, runner: CliRunner):
        """Test stop --help."""
        result = runner.invoke(app, ["stop", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--force" in result.output
        assert "--timeout" in result.output

    def test_stop_nonexistent_agent(self, runner: CliRunner):
        """Test stop with nonexistent agent."""
        result = runner.invoke(app, ["stop", "nonexistent_agent_12345"])
        # Should fail gracefully - not crash
        assert result.exit_code in [0, 1]

    def test_stop_no_args(self, runner: CliRunner):
        """Test stop without arguments."""
        result = runner.invoke(app, ["stop"])
        # Should show usage or error
        assert result.exit_code in [0, 1, 2]

    def test_stop_all_option(self, runner: CliRunner):
        """Test stop --all option is documented."""
        result = runner.invoke(app, ["stop", "--help"])
        assert "--all" in result.output
        assert "-a" in result.output

    def test_stop_force_option(self, runner: CliRunner):
        """Test stop --force option is documented."""
        result = runner.invoke(app, ["stop", "--help"])
        assert "--force" in result.output
        assert "-f" in result.output
