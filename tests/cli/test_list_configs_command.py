from typer.testing import CliRunner

from cli import app

runner = CliRunner()


class TestListConfigsCommand:

    def test_list_configs_runs(self):
        result = runner.invoke(app, ["list-configs"])

        assert result.exit_code == 0

    def test_list_configs_shows_configs(self):
        result = runner.invoke(app, ["list-configs"])

        assert result.exit_code == 0
        assert "test" in result.stdout.lower() or "spot" in result.stdout.lower()

    def test_list_configs_shows_header(self):
        result = runner.invoke(app, ["list-configs"])

        assert result.exit_code == 0
        assert (
            "configuration" in result.stdout.lower()
            or "config" in result.stdout.lower()
        )
