import os
import shutil
import tempfile
from typing import Generator

import pytest

from cli import _resolve_config_path


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


class TestResolveConfigPath:

    def test_absolute_path_exists(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "test.json5")
        with open(config_path, "w") as f:
            f.write("{}")

        result = _resolve_config_path(config_path)
        assert result == config_path

    def test_path_without_extension(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "test.json5")
        with open(config_path, "w") as f:
            f.write("{}")

        path_without_ext = os.path.join(temp_dir, "test")
        result = _resolve_config_path(path_without_ext)
        assert result == config_path

    def test_config_dir_lookup(self):
        result = _resolve_config_path("test")
        assert result.endswith("test.json5")
        assert os.path.exists(result)

    def test_config_dir_lookup_with_extension(self):
        result = _resolve_config_path("test.json5")
        assert result.endswith("test.json5")
        assert os.path.exists(result)

    def test_nonexistent_config_raises_error(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            _resolve_config_path("nonexistent_config_12345")

        assert "nonexistent_config_12345" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_returns_absolute_path(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "test.json5")
        with open(config_path, "w") as f:
            f.write("{}")

        result = _resolve_config_path(config_path)
        assert os.path.isabs(result)

    def test_relative_path_converted_to_absolute(self, temp_dir: str):
        config_path = os.path.join(temp_dir, "myconfig.json5")
        with open(config_path, "w") as f:
            f.write("{}")

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = _resolve_config_path("myconfig.json5")
            assert os.path.isabs(result)
            assert os.path.realpath(result) == os.path.realpath(config_path)
        finally:
            os.chdir(original_cwd)
