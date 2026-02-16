import json
import os
import subprocess
from pathlib import Path


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _group_descriptor_counts(descriptor_info):
    grouped = {}
    for record in descriptor_info:
        site = record.get("metal_site", {})
        key = (
            site.get("chain"),
            site.get("residue"),
            site.get("sequence"),
            site.get("icode"),
            site.get("altloc"),
            site.get("metal"),
            site.get("metalElement"),
        )
        grouped[key] = grouped.get(key, 0) + 1
    return grouped


def test_stats_debug_sidecars_and_schema(tmp_path: Path, cli_cmd):
    model = Path("tests/data/models/3kw8.cif").resolve()
    output = tmp_path / "cu_stats.json"
    output_plain = tmp_path / "cu_stats_plain.json"

    _run(
        cli_cmd(
            "--no-progress",
            "stats",
            "--debug",
            "--debug-level",
            "detailed",
            "-l",
            "CU",
            "-p",
            str(model),
            "-o",
            str(output),
        )
    )
    _run(
        cli_cmd(
            "--no-progress",
            "stats",
            "-l",
            "CU",
            "-p",
            str(model),
            "-o",
            str(output_plain),
        )
    )

    debug_json = Path(str(output) + ".debug.json")
    debug_md = Path(str(output) + ".debug.md")
    assert output.exists()
    assert debug_json.exists()
    assert debug_md.exists()

    debug_payload = _load_json(debug_json)
    assert "domain_report" in debug_payload
    assert "descriptor_info" in debug_payload
    assert "trace" in debug_payload
    assert "logs" in debug_payload

    step_names = [step["name"] for step in debug_payload["domain_report"].get("steps", [])]
    assert "Linear Descriptor Generation" in step_names

    descriptor = debug_payload["descriptor_info"]
    assert descriptor, "descriptor_info should not be empty"
    first = descriptor[0]
    for field in (
        "metal_site",
        "class",
        "class_code",
        "descriptor",
        "ordered_elements",
        "index_mapping",
        "atom_names_with_symmetries",
        "element_names",
    ):
        assert field in first
    assert all(isinstance(v, int) and v >= 0 for v in first["index_mapping"])

    # Main output should remain unchanged with debug enabled.
    assert _load_json(output) == _load_json(output_plain)


def test_stats_debug_level_descriptor_scope(tmp_path: Path, cli_cmd):
    model = Path("tests/data/models/3kw8.cif").resolve()
    outputs = {
        "summary": tmp_path / "summary.json",
        "detailed": tmp_path / "detailed.json",
        "max": tmp_path / "max.json",
    }

    for level, output in outputs.items():
        _run(
            cli_cmd(
                "--no-progress",
                "stats",
                "--debug",
                "--debug-level",
                level,
                "-l",
                "CU",
                "-p",
                str(model),
                "-o",
                str(output),
            )
        )

    summary = _group_descriptor_counts(
        _load_json(Path(str(outputs["summary"]) + ".debug.json"))["descriptor_info"]
    )
    detailed = _group_descriptor_counts(
        _load_json(Path(str(outputs["detailed"]) + ".debug.json"))["descriptor_info"]
    )
    max_level = _group_descriptor_counts(
        _load_json(Path(str(outputs["max"]) + ".debug.json"))["descriptor_info"]
    )

    all_keys = set(summary) | set(detailed) | set(max_level)
    for key in all_keys:
        s = summary.get(key, 0)
        d = detailed.get(key, 0)
        m = max_level.get(key, 0)
        assert s <= 1
        assert d <= 4
        assert s <= d <= m


def test_stats_debug_no_metal_flag(tmp_path: Path, cli_cmd):
    pdb_path = tmp_path / "no_metal.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   2.321  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.305   2.445  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "no_metal.json"
    _run(
        cli_cmd(
            "--no-progress",
            "stats",
            "--debug",
            "-p",
            str(pdb_path),
            "-o",
            str(output),
        )
    )

    status = _load_json(Path(str(output) + ".status.json"))
    assert status["status"] == "Failure"

    debug_json = Path(str(output) + ".debug.json")
    assert debug_json.exists()
    debug_payload = _load_json(debug_json)
    assert "no metal found" in debug_payload.get("domain_report", {}).get("flags", [])


def test_stats_multi_ligand_writes_per_output_sidecars(tmp_path: Path, cli_cmd):
    model = Path("tests/data/models/4dl8.cif").resolve()
    output_dir = tmp_path / "all_ligands"
    output_dir.mkdir(parents=True, exist_ok=True)

    _run(
        cli_cmd(
            "--no-progress",
            "stats",
            "--debug",
            "-p",
            str(model),
            "-o",
            str(output_dir),
        )
    )

    main_outputs = [
        p
        for p in output_dir.glob("*.json")
        if not p.name.endswith(".metal_metal.json")
        and not p.name.endswith(".debug.json")
        and not p.name.endswith(".status.json")
    ]
    assert main_outputs, "Expected at least one ligand output file"

    for output in main_outputs:
        assert Path(str(output) + ".debug.json").exists()
        assert Path(str(output) + ".debug.md").exists()
