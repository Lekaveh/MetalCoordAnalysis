import re
import subprocess

def test_help(cli_cmd):
    """Test basic CLI functionality."""
    result = subprocess.run(
        cli_cmd("--help"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, "CLI help command failed."
    assert "usage" in result.stdout.lower(), "Help output missing 'usage' information."


def test_version(cli_cmd):
    """Test CLI version flag."""
    result = subprocess.run(
        cli_cmd("--version"),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0, "CLI version command failed."
    assert re.search(r"metalCoord \d+\.\d+\.\d+",
                     result.stdout), "Version output does not contain \"metalCoord\" or a version number."
