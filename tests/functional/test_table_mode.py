import json
import subprocess
from pathlib import Path


def _run(cmd, cwd: Path):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )


def test_table_default_json_output(tmp_path: Path, cli_cmd):
    result = _run(cli_cmd("--no-progress", "table"), tmp_path)
    assert result.returncode == 0

    output_json = tmp_path / "metalcoord_table.json"
    assert output_json.exists(), "Expected default JSON table output in current directory."

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload, "Expected non-empty class table payload."

    first_row = payload[0]
    assert {"class_name", "class_code", "atom_coordinates"} <= set(first_row.keys())
    assert isinstance(first_row["atom_coordinates"], list)
    assert first_row["atom_coordinates"], "Expected atom coordinate rows."
    assert {"atom_index", "x", "y", "z"} <= set(first_row["atom_coordinates"][0].keys())

    linear = next((row for row in payload if row["class_name"] == "linear"), None)
    assert linear is not None
    assert linear["class_code"] == "LIN"


def test_table_html_output(tmp_path: Path, cli_cmd):
    output_dir = tmp_path / "tables"
    result = _run(
        cli_cmd(
            "--no-progress",
            "table",
            "-o",
            str(output_dir),
            "--format",
            "html",
        ),
        tmp_path,
    )
    assert result.returncode == 0

    output_html = output_dir / "metalcoord_table.html"
    assert output_html.exists(), "Expected HTML table output."
    assert not (output_dir / "metalcoord_table.json").exists()

    html = output_html.read_text(encoding="utf-8")
    assert "<table>" in html
    assert "Class Name" in html


def test_table_both_output_formats(tmp_path: Path, cli_cmd):
    output_dir = tmp_path / "tables"
    result = _run(
        cli_cmd(
            "--no-progress",
            "table",
            "-o",
            str(output_dir),
            "--format",
            "both",
        ),
        tmp_path,
    )
    assert result.returncode == 0
    assert (output_dir / "metalcoord_table.json").exists()
    assert (output_dir / "metalcoord_table.html").exists()
