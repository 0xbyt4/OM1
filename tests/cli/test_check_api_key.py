import os
from io import StringIO
from unittest.mock import patch

from cli import _check_api_key


class TestCheckApiKey:

    def test_no_api_key_shows_warning(self):
        config = {"api_key": ""}

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OM_API_KEY", None)

            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=False)
                output = mock_stdout.getvalue()

            assert "Warning" in output or "warning" in output.lower()
            assert "No API key" in output or "API key" in output

    def test_openmind_free_key_shows_warning(self):
        config = {"api_key": "openmind_free"}

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OM_API_KEY", None)

            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=False)
                output = mock_stdout.getvalue()

            assert "Warning" in output or "API key" in output

    def test_valid_api_key_no_warning(self):
        config = {"api_key": "my_custom_api_key"}

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OM_API_KEY", None)

            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=False)
                output = mock_stdout.getvalue()

            assert "Warning" not in output

    def test_env_api_key_no_warning(self):
        config = {"api_key": ""}

        with patch.dict(os.environ, {"OM_API_KEY": "env_api_key"}, clear=False):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=False)
                output = mock_stdout.getvalue()

            assert "Warning" not in output

    def test_verbose_shows_configured_message(self):
        config = {"api_key": "my_api_key"}

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OM_API_KEY", None)

            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=True)
                output = mock_stdout.getvalue()

            assert "configured" in output.lower()

    def test_verbose_env_shows_from_environment(self):
        config = {"api_key": ""}

        with patch.dict(os.environ, {"OM_API_KEY": "env_key"}, clear=False):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                _check_api_key(config, verbose=True)
                output = mock_stdout.getvalue()

            assert "environment" in output.lower()
