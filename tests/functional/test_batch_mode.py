import json
import shutil
import subprocess
import textwrap
from pathlib import Path


def _run(cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _prepare_input_file(tmp_path: Path, source: str, rel_dest: str) -> Path:
    source_path = Path(source).resolve()
    dest_path = tmp_path / rel_dest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)
    return dest_path


def test_batch_help(cli_cmd):
    result = _run(cli_cmd("batch", "--help"))
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout


def test_batch_mixed_modes_and_relative_paths(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/3kw8.cif", "inputs/3kw8.cif")
    _prepare_input_file(tmp_path, "tests/data/dicts/SF4.cif", "dicts/SF4.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        defaults:
          debug: false
        stats:
          jobs:
            - name: cu_stats
              ligand: CU
              pdb: ./inputs/3kw8.cif
        update:
          defaults:
            cif: true
          jobs:
            - name: sf4_update
              input: ./dicts/SF4.cif
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path)))
    assert result.returncode == 0

    stats_output = tmp_path / "out" / "stats" / "3kw8_CU.json"
    update_output = tmp_path / "out" / "update" / "SF4.cif"
    report_json = tmp_path / "out" / "batch_report.json"
    report_md = tmp_path / "out" / "batch_report.md"
    assert stats_output.exists()
    assert update_output.exists()
    assert report_json.exists()
    assert report_md.exists()

    report = _load_json(report_json)
    assert report["status"] == "success"
    assert report["meta"]["total"] == 2
    assert report["meta"]["failure_count"] == 0
    assert all(job["status"] == "success" for job in report["jobs"])
    assert str(stats_output) == report["jobs"][0]["resolved_output"]
    assert str(update_output) == report["jobs"][1]["resolved_output"]


def test_batch_multi_ligand_stats_auto_directory(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/4dl8.cif", "inputs/4dl8.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        stats:
          jobs:
            - name: all_ligands
              pdb: ./inputs/4dl8.cif
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path)))
    assert result.returncode == 0

    output_dir = tmp_path / "out" / "stats" / "4dl8"
    assert output_dir.is_dir()
    main_outputs = [
        p
        for p in output_dir.glob("*.json")
        if not p.name.endswith(".metal_metal.json")
        and not p.name.endswith(".debug.json")
        and not p.name.endswith(".status.json")
    ]
    assert main_outputs


def test_batch_continue_on_error_and_exit_code(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/3kw8.cif", "inputs/3kw8.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        stats:
          jobs:
            - name: broken
              ligand: CU
              pdb: ./inputs/not_found.cif
            - name: valid
              ligand: CU
              pdb: ./inputs/3kw8.cif
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path)), check=False)
    assert result.returncode == 1

    valid_output = tmp_path / "out" / "stats" / "3kw8_CU.json"
    assert valid_output.exists()
    report = _load_json(tmp_path / "out" / "batch_report.json")
    assert report["status"] == "partial_failure"
    assert report["meta"]["failure_count"] == 1
    assert report["meta"]["success_count"] == 1
    assert report["jobs"][0]["status"] == "failure"
    assert report["jobs"][1]["status"] == "success"


def test_batch_dry_run_writes_report_only(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/3kw8.cif", "inputs/3kw8.cif")
    _prepare_input_file(tmp_path, "tests/data/dicts/SF4.cif", "dicts/SF4.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        stats:
          jobs:
            - ligand: CU
              pdb: ./inputs/3kw8.cif
        update:
          defaults:
            cif: true
          jobs:
            - input: ./dicts/SF4.cif
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path), "--dry-run"))
    assert result.returncode == 0

    report_json = tmp_path / "out" / "batch_report.json"
    assert report_json.exists()
    report = _load_json(report_json)
    assert report["status"] == "dry_run"
    assert all(job["status"] == "dry_run" for job in report["jobs"])

    assert not (tmp_path / "out" / "stats" / "3kw8_CU.json").exists()
    assert not (tmp_path / "out" / "update" / "SF4.cif").exists()


def test_batch_validation_error_exit_two(tmp_path: Path, cli_cmd):
    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        unknown_key: true
        stats:
          jobs:
            - ligand: CU
              pdb: ./inputs/3kw8.cif
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path)), check=False)
    assert result.returncode == 2
    assert not (tmp_path / "out" / "batch_report.json").exists()


def test_batch_detects_output_collisions_in_dry_run(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/3kw8.cif", "inputs/3kw8.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        stats:
          jobs:
            - ligand: CU
              pdb: ./inputs/3kw8.cif
              output: ./out/stats/same.json
            - ligand: CU
              pdb: ./inputs/3kw8.cif
              output: ./out/stats/same.json
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path), "--dry-run"), check=False)
    assert result.returncode == 2
    assert not (tmp_path / "out" / "batch_report.json").exists()


def test_batch_debug_defaults_with_job_override(tmp_path: Path, cli_cmd):
    _prepare_input_file(tmp_path, "tests/data/models/3kw8.cif", "inputs/3kw8.cif")

    config_path = tmp_path / "batch.yaml"
    _write_yaml(
        config_path,
        """
        version: 1
        output_root: ./out
        defaults:
          debug: true
          debug_level: summary
        stats:
          jobs:
            - name: with_debug
              ligand: CU
              pdb: ./inputs/3kw8.cif
              output: ./out/stats/with_debug.json
            - name: no_debug
              ligand: CU
              pdb: ./inputs/3kw8.cif
              output: ./out/stats/no_debug.json
              debug: false
        """,
    )

    result = _run(cli_cmd("--no-progress", "batch", "-f", str(config_path)))
    assert result.returncode == 0

    with_debug = tmp_path / "out" / "stats" / "with_debug.json"
    no_debug = tmp_path / "out" / "stats" / "no_debug.json"
    assert with_debug.exists()
    assert no_debug.exists()
    assert Path(str(with_debug) + ".debug.json").exists()
    assert Path(str(with_debug) + ".debug.md").exists()
    assert not Path(str(no_debug) + ".debug.json").exists()
    assert not Path(str(no_debug) + ".debug.md").exists()
