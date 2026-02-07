import json
import subprocess
from pathlib import Path


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_update_debug_sidecars_and_schema(tmp_path: Path, cli_cmd):
    input_cif = Path("tests/data/dicts/SF4.cif").resolve()
    output_cif = tmp_path / "SF4_out.cif"

    _run(
        cli_cmd(
            "--no-progress",
            "update",
            "--debug",
            "--debug-level",
            "detailed",
            "--cif",
            "-i",
            str(input_cif),
            "-o",
            str(output_cif),
        )
    )

    debug_json = Path(str(output_cif) + ".debug.json")
    debug_md = Path(str(output_cif) + ".debug.md")
    assert output_cif.exists()
    assert debug_json.exists()
    assert debug_md.exists()

    payload = _load_json(debug_json)
    step_names = [step["name"] for step in payload.get("domain_report", {}).get("steps", [])]
    assert "Linear Descriptor Generation" in step_names
    assert "descriptor_info" in payload
    assert "trace" in payload
    assert "logs" in payload


def test_update_debug_no_metal_case(tmp_path: Path, cli_cmd):
    input_cif = tmp_path / "NOM.cif"
    input_cif.write_text(
        "\n".join(
            [
                "data_comp_NOM",
                "loop_",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "C1 C 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_cif = tmp_path / "NOM_out.cif"
    _run(
        cli_cmd(
            "--no-progress",
            "update",
            "--debug",
            "--cif",
            "-i",
            str(input_cif),
            "-o",
            str(output_cif),
        )
    )

    debug_json = Path(str(output_cif) + ".debug.json")
    assert debug_json.exists()
    payload = _load_json(debug_json)
    assert "no metal found" in payload.get("domain_report", {}).get("flags", [])
