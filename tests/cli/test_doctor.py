"""Tests for CLI doctor command."""

from typer.testing import CliRunner

from cli.main import app


class TestDoctorCommand:
    """Tests for doctor command."""

    def test_doctor_help(self, runner: CliRunner):
        """Test doctor --help."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--verbose" in result.output

    def test_doctor_basic(self, runner: CliRunner):
        """Test basic doctor command."""
        result = runner.invoke(app, ["doctor"])
        # Should run without error (may have warnings but not crash)
        assert result.exit_code in [0, 1]
        assert "Python Version" in result.output

    def test_doctor_verbose(self, runner: CliRunner):
        """Test doctor --verbose shows details."""
        result = runner.invoke(app, ["doctor", "--verbose"])
        assert result.exit_code in [0, 1]
        # Verbose should show path details
        assert "Path:" in result.output or "python" in result.output.lower()

    def test_doctor_config_option(self, runner: CliRunner):
        """Test doctor --config option."""
        result = runner.invoke(app, ["doctor", "--config", "ollama"])
        assert result.exit_code in [0, 1]
        assert "ollama" in result.output.lower() or "Ollama" in result.output
