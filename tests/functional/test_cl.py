import re
import subprocess


def test_help():
    """Test basic CLI functionality."""
    result = subprocess.run(
        ["metalCoord", "--help"], capture_output=True, text=True, check=True, shell = True
    )
    assert result.returncode == 0, "CLI help command failed."
    assert "usage" in result.stdout.lower(), "Help output missing 'usage' information."


def test_version():
    """Test CLI version flag."""
    result = subprocess.run(
        ["metalCoord", "--version"], capture_output=True, text=True, check=True, shell = True
    )

    assert result.returncode == 0, "CLI version command failed."
    assert re.search(r"metalCoord \d+\.\d+\.\d+",
                     result.stdout), "Version output does not contain \"metalCoord\" or a version number."
