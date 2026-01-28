import os
import shutil
import tempfile
from typing import Generator

import pytest

from cli import _check_class_in_dir


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


class TestCheckClassInDir:

    def test_classdef_found(self, temp_dir: str):
        with open(os.path.join(temp_dir, "test_classdef.py"), "w") as f:
            f.write("class MyClass:\n    pass\n")

        assert _check_class_in_dir(temp_dir, "MyClass") is True

    def test_classdef_not_found(self, temp_dir: str):
        with open(os.path.join(temp_dir, "test_classdef.py"), "w") as f:
            f.write("class MyClass:\n    pass\n")

        assert _check_class_in_dir(temp_dir, "NonExistent") is False

    def test_empty_directory(self, temp_dir: str):
        assert _check_class_in_dir(temp_dir, "AnyClass") is False

    def test_nonexistent_directory(self):
        assert _check_class_in_dir("/nonexistent/path", "AnyClass") is False

    def test_init_file_ignored(self, temp_dir: str):
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("class InitClass:\n    pass\n")

        assert _check_class_in_dir(temp_dir, "InitClass") is False

    def test_multiple_files(self, temp_dir: str):
        with open(os.path.join(temp_dir, "file1.py"), "w") as f:
            f.write("class ClassOne:\n    pass\n")
        with open(os.path.join(temp_dir, "file2.py"), "w") as f:
            f.write("class ClassTwo:\n    pass\n")

        assert _check_class_in_dir(temp_dir, "ClassOne") is True
        assert _check_class_in_dir(temp_dir, "ClassTwo") is True
        assert _check_class_in_dir(temp_dir, "ClassThree") is False

    def test_syntax_error_file_skipped(self, temp_dir: str):
        with open(os.path.join(temp_dir, "bad_syntax.py"), "w") as f:
            f.write("class Broken\n    pass\n")
        with open(os.path.join(temp_dir, "good.py"), "w") as f:
            f.write("class GoodClass:\n    pass\n")

        assert _check_class_in_dir(temp_dir, "GoodClass") is True

    def test_nested_class_not_found(self, temp_dir: str):
        with open(os.path.join(temp_dir, "nested.py"), "w") as f:
            f.write("class Outer:\n    class Inner:\n        pass\n")

        assert _check_class_in_dir(temp_dir, "Outer") is True
        assert _check_class_in_dir(temp_dir, "Inner") is False

    def test_class_with_inheritance(self, temp_dir: str):
        with open(os.path.join(temp_dir, "inherited.py"), "w") as f:
            f.write("class MyPlugin(BasePlugin):\n    pass\n")

        assert _check_class_in_dir(temp_dir, "MyPlugin") is True

    def test_non_py_files_ignored(self, temp_dir: str):
        with open(os.path.join(temp_dir, "config.txt"), "w") as f:
            f.write("class FakeClass:\n    pass\n")
        with open(os.path.join(temp_dir, "data.json"), "w") as f:
            f.write('{"class": "NotAClass"}')

        assert _check_class_in_dir(temp_dir, "FakeClass") is False
        assert _check_class_in_dir(temp_dir, "NotAClass") is False
