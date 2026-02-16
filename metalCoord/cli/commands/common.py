import json
import os
from pathlib import Path
from typing import Any, Optional

from metalCoord.config import Config
from metalCoord.debug import DebugRecorder, resolve_debug_paths
from metalCoord.debug_domain import render_domain_markdown
from metalCoord.logging import Logger


def configure_statistics(args) -> None:
    if args.min_size > args.max_size:
        raise ValueError("Minimum sample size must be less or equal than maximum sample size.")

    Config().ideal_angles = getattr(args, "ideal_angles", False)
    Config().distance_threshold = args.dist
    Config().procrustes_threshold = args.threshold
    Config().min_sample_size = args.min_size
    Config().simple = getattr(args, "simple", False)
    Config().save = getattr(args, "save", False)
    Config().use_pdb = getattr(args, "use_pdb", False)
    Config().output_folder = os.path.abspath(os.path.dirname(args.output))
    Config().output_file = os.path.basename(args.output)
    Config().max_coordination_number = args.coordination
    Config().max_sample_size = args.max_size


def configure_debug(args, command: str) -> None:
    Config().debug = getattr(args, "debug", False)
    Config().debug_level = getattr(args, "debug_level", "detailed")
    Config().debug_output = getattr(args, "debug_output", None)
    Config().debug_command = command
    Config().debug_recorder = None
    Config().debug_written = False
    Config().debug_log_mark = 0
    if Config().debug:
        Logger().enable_capture(True, reset=True)
        Config().debug_log_mark = Logger().mark()
    else:
        Logger().enable_capture(False)


def merge_settings(
    global_defaults: Optional[dict[str, Any]],
    mode_defaults: Optional[dict[str, Any]],
    job: Optional[dict[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if global_defaults:
        merged.update(global_defaults)
    if mode_defaults:
        merged.update(mode_defaults)
    if job:
        merged.update(job)
    return merged


def _fallback_debug_paths(output: str, override: Optional[str], multi: bool) -> tuple[str, str]:
    return resolve_debug_paths(output, override, multi_ligand=multi)


def write_failure_debug_report(args, command: str, reason: str) -> None:
    if not Config().debug or Config().debug_written:
        return

    existing = Config().debug_recorder
    if existing:
        recorder = existing
    else:
        recorder = DebugRecorder(command, Config().debug_level)
        output_path = getattr(args, "output", "")
        if command == "stats" and not getattr(args, "ligand", "") and not Path(output_path).suffix:
            output_path = os.path.join(output_path, "analysis")
        json_path, md_path = _fallback_debug_paths(
            output_path, Config().debug_output, command == "stats" and not getattr(args, "ligand", "")
        )
        recorder.set_paths(json_path, md_path)
        recorder.set_inputs(
            {
                "source": getattr(args, "pdb", getattr(args, "input", None)),
                "ligand": getattr(args, "ligand", None),
                "pdb": getattr(args, "pdb", None),
                "class": getattr(args, "cl", None),
                "thresholds": {
                    "distance": Config().distance_threshold,
                    "procrustes": Config().procrustes_threshold,
                    "min_sample_size": Config().min_sample_size,
                    "metal_distance": Config().metal_distance_threshold,
                },
            }
        )
        recorder.set_outputs(
            {
                "main_output": getattr(args, "output", None),
            }
        )
        recorder.set_log_mark(Config().debug_log_mark)

    recorder.set_status("failure")
    recorder.add_error(reason)
    recorder.set_logs(Logger().records_since(recorder.log_mark))
    if not recorder.payload.get("domain_report"):
        flag = "analysis failure"
        if "No metal-containing ligands found" in reason or "No metal found" in reason:
            flag = "no metal found"
        recorder.set_domain_report(
            {
                "title": "Metal Coordination Analysis Report",
                "inputs": recorder.payload.get("inputs", {}),
                "steps": [],
                "summary": {},
                "flags": [flag],
            }
        )
    recorder.payload["domain_report"]["markdown"] = render_domain_markdown(
        recorder.payload.get("domain_report", {}),
        recorder.payload.get("logs", []),
    )
    recorder.write_json()
    recorder.write_markdown(
        render_domain_markdown(recorder.payload.get("domain_report", {}), recorder.payload.get("logs", []))
    )
    Config().debug_recorder = None
    Config().debug_written = True


def write_status(status: str, reason: Optional[str] = None, ensure_dir: bool = False) -> None:
    if ensure_dir:
        Path(Config().output_folder).mkdir(exist_ok=True, parents=True)
    status_path = os.path.join(Config().output_folder, Config().output_file + ".status.json")
    payload = {"status": status}
    if reason:
        payload["Reason"] = reason
    with open(status_path, 'w', encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4, separators=(',', ': '))


def status_path_for_output(output_path: str) -> str:
    output_folder = os.path.abspath(os.path.dirname(output_path))
    output_file = os.path.basename(output_path)
    return os.path.join(output_folder, output_file + ".status.json")


def read_status_for_output(output_path: str) -> dict[str, Any]:
    status_path = status_path_for_output(output_path)
    if not os.path.exists(status_path):
        return {
            "status": "Failure",
            "Reason": f"Status file not found: {status_path}",
        }
    with open(status_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def log_exception(exc: Exception) -> None:
    Logger().error(f"{str(exc)}")
