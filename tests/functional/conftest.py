import sys

import pytest


@pytest.fixture
def cli_cmd():
    """Build a cross-platform CLI command using the current Python interpreter."""
    def _cmd(*args: str):
        return [sys.executable, "-m", "metalCoord.run", *args]

    return _cmd
