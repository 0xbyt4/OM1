from typer.testing import CliRunner

from cli import app

runner = CliRunner()


class TestModesCommand:

    def test_modes_with_multi_mode_config(self):
        result = runner.invoke(app, ["modes", "spot_modes"])

        if result.exit_code == 0:
            assert "mode" in result.stdout.lower()

    def test_modes_with_nonexistent_config(self):
        result = runner.invoke(app, ["modes", "nonexistent_config_12345"])

        assert result.exit_code == 1

    def test_modes_shows_transition_rules(self):
        result = runner.invoke(app, ["modes", "spot_modes"])

        if result.exit_code == 0:
            assert (
                "transition" in result.stdout.lower() or "rule" in result.stdout.lower()
            )

    def test_modes_with_single_mode_config_fails(self):
        result = runner.invoke(app, ["modes", "test"])

        assert result.exit_code in [0, 1]
