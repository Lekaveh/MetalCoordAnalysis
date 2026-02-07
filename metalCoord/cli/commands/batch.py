import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from metalCoord.cli.commands.common import (
    merge_settings,
    read_status_for_output,
    status_path_for_output,
)
from metalCoord.cli.commands.stats import handle_stats
from metalCoord.cli.commands.update import handle_update
from metalCoord.logging import Logger


class BatchConfigError(ValueError):
    pass


TOP_LEVEL_KEYS = {
    "version",
    "output_root",
    "stats_output_dir",
    "update_output_dir",
    "defaults",
    "stats",
    "update",
}

COMMON_OPTION_KEYS = {
    "dist",
    "threshold",
    "min_size",
    "max_size",
    "coordination",
    "ideal_angles",
    "simple",
    "save",
    "use_pdb",
    "cl",
    "debug",
    "debug_level",
    "debug_output",
}

STATS_OPTION_KEYS = COMMON_OPTION_KEYS | {
    "name",
    "ligand",
    "pdb",
    "metal_distance",
    "output",
}

UPDATE_OPTION_KEYS = COMMON_OPTION_KEYS | {
    "name",
    "input",
    "pdb",
    "cif",
    "output",
}

STATS_DEFAULTS = {
    "dist": 0.5,
    "threshold": 0.3,
    "min_size": 30,
    "max_size": 2000,
    "coordination": 1000,
    "ideal_angles": False,
    "simple": False,
    "save": False,
    "use_pdb": False,
    "cl": None,
    "debug": False,
    "debug_level": "detailed",
    "debug_output": None,
    "metal_distance": 0.3,
    "ligand": "",
}

UPDATE_DEFAULTS = {
    "dist": 0.5,
    "threshold": 0.3,
    "min_size": 30,
    "max_size": 2000,
    "coordination": 1000,
    "ideal_angles": False,
    "simple": False,
    "save": False,
    "use_pdb": False,
    "cl": None,
    "debug": False,
    "debug_level": "detailed",
    "debug_output": None,
    "cif": False,
    "pdb": None,
}

STATS_SECTION_KEYS = {"defaults", "jobs"}
UPDATE_SECTION_KEYS = {"defaults", "jobs"}
STATS_DEFAULT_KEYS = COMMON_OPTION_KEYS | {"metal_distance"}
UPDATE_DEFAULT_KEYS = COMMON_OPTION_KEYS | {"cif"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BatchConfigError(f"{context} must be a mapping.")
    return value


def _ensure_list(value: Any, context: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise BatchConfigError(f"{context} must be a list.")
    return value


def _unknown_keys(allowed: set[str], values: dict[str, Any], context: str) -> None:
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        raise BatchConfigError(f"Unknown keys in {context}: {', '.join(unknown)}")


def _resolve_path(value: str, base_dir: Path) -> str:
    if os.path.isabs(value):
        return str(Path(value))
    return str((base_dir / value).resolve())


def _looks_like_path(value: str) -> bool:
    path_markers = ("/", "\\", "./", "../")
    if value.startswith(path_markers):
        return True
    if "/" in value or "\\" in value:
        return True
    return Path(value).suffix.lower() in {".pdb", ".cif", ".mmcif", ".ent", ".gz"}


def _normalize_optional_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    return _resolve_path(value, base_dir)


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchConfigError(f"{context} must be a non-empty string.")
    return value


def _validate_number_in_range(name: str, value: Any, low: float, high: float, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BatchConfigError(f"{context}.{name} must be a number in [{low}, {high}].")
    number = float(value)
    if number < low or number > high:
        raise BatchConfigError(f"{context}.{name} must be in [{low}, {high}].")
    return number


def _validate_positive_int(name: str, value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BatchConfigError(f"{context}.{name} must be a positive integer.")
    return value


def _validate_coordination(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
        raise BatchConfigError(f"{context}.coordination must be an integer greater than 1.")
    return value


def _validate_boolean(name: str, value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise BatchConfigError(f"{context}.{name} must be a boolean.")
    return value


def _validate_debug_level(value: Any, context: str) -> str:
    if value not in {"summary", "detailed", "max"}:
        raise BatchConfigError(f"{context}.debug_level must be one of: summary, detailed, max.")
    return str(value)


def _validate_effective_common(values: dict[str, Any], context: str) -> dict[str, Any]:
    values["dist"] = _validate_number_in_range("dist", values["dist"], 0, 1, context)
    values["threshold"] = _validate_number_in_range("threshold", values["threshold"], 0, 1, context)
    values["min_size"] = _validate_positive_int("min_size", values["min_size"], context)
    values["max_size"] = _validate_positive_int("max_size", values["max_size"], context)
    if values["min_size"] > values["max_size"]:
        raise BatchConfigError(f"{context}.min_size must be less or equal than {context}.max_size.")
    values["coordination"] = _validate_coordination(values["coordination"], context)
    values["ideal_angles"] = _validate_boolean("ideal_angles", values["ideal_angles"], context)
    values["simple"] = _validate_boolean("simple", values["simple"], context)
    values["save"] = _validate_boolean("save", values["save"], context)
    values["use_pdb"] = _validate_boolean("use_pdb", values["use_pdb"], context)
    values["debug"] = _validate_boolean("debug", values["debug"], context)
    values["debug_level"] = _validate_debug_level(values["debug_level"], context)

    cl_value = values.get("cl")
    if cl_value is not None and not isinstance(cl_value, str):
        raise BatchConfigError(f"{context}.cl must be a string or null.")

    debug_output = values.get("debug_output")
    if debug_output is not None and not isinstance(debug_output, str):
        raise BatchConfigError(f"{context}.debug_output must be a string or null.")
    return values


def _load_yaml(config_path: Path) -> dict[str, Any]:
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BatchConfigError(f"Cannot read config file {config_path}: {exc}") from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise BatchConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise BatchConfigError("Batch config must be a YAML mapping.")
    return payload


def _build_stats_args(options: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        ligand=options.get("ligand", ""),
        pdb=options["pdb"],
        output=options["output"],
        dist=options["dist"],
        threshold=options["threshold"],
        min_size=options["min_size"],
        max_size=options["max_size"],
        ideal_angles=options["ideal_angles"],
        simple=options["simple"],
        save=options["save"],
        use_pdb=options["use_pdb"],
        coordination=options["coordination"],
        cl=options.get("cl"),
        metal_distance=options["metal_distance"],
        debug=options["debug"],
        debug_level=options["debug_level"],
        debug_output=options.get("debug_output"),
    )


def _build_update_args(options: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        input=options["input"],
        output=options["output"],
        pdb=options.get("pdb"),
        dist=options["dist"],
        threshold=options["threshold"],
        min_size=options["min_size"],
        max_size=options["max_size"],
        ideal_angles=options["ideal_angles"],
        simple=options["simple"],
        save=options["save"],
        use_pdb=options["use_pdb"],
        coordination=options["coordination"],
        cl=options.get("cl"),
        cif=options["cif"],
        debug=options["debug"],
        debug_level=options["debug_level"],
        debug_output=options.get("debug_output"),
    )


def _render_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# MetalCoord Batch Report",
        "",
        "## Summary",
        f"- Config: `{meta['config']}`",
        f"- Status: `{payload['status']}`",
        f"- Dry run: `{meta['dry_run']}`",
        f"- Total jobs: `{meta['total']}`",
        f"- Success: `{meta['success_count']}`",
        f"- Failure: `{meta['failure_count']}`",
        f"- Started: `{meta['started_at']}`",
        f"- Finished: `{meta['finished_at']}`",
        "",
        "## Jobs",
        "| # | Mode | Name | Status | Output | Duration (s) |",
        "|---|------|------|--------|--------|--------------|",
    ]

    for job in payload["jobs"]:
        lines.append(
            f"| {job['index']} | {job['mode']} | {job['name']} | {job['status']} | "
            f"`{job['resolved_output']}` | {job.get('duration_sec', 0)} |"
        )

    failures = [job for job in payload["jobs"] if job["status"] == "failure"]
    lines.extend(["", "## Failure Reasons"])
    if failures:
        for job in failures:
            lines.append(
                f"- Job `{job['index']}` (`{job['name']}`): {job.get('reason', 'Unknown failure')}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Batch JSON: `{meta['report_json']}`",
            f"- Batch Markdown: `{meta['report_md']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_batch_reports(output_root: Path, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "batch_report.json"
    md_path = output_root / "batch_report.md"
    payload["meta"]["report_json"] = str(json_path)
    payload["meta"]["report_md"] = str(md_path)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, separators=(",", ": "))
    md_path.write_text(_render_markdown_report(payload), encoding="utf-8")


def _resolve_pdb_value(value: str, base_dir: Path) -> str:
    if _looks_like_path(value):
        return _resolve_path(value, base_dir)
    return value


def _resolve_output_path(value: str, base_dir: Path) -> str:
    return _resolve_path(value, base_dir)


def _plan_jobs(config_payload: dict[str, Any], config_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    _unknown_keys(TOP_LEVEL_KEYS, config_payload, "top-level config")
    version = config_payload.get("version", 1)
    if version != 1:
        raise BatchConfigError(f"Unsupported config version: {version}. Expected 1.")

    config_dir = config_path.parent
    output_root_value = config_payload.get("output_root")
    output_root_str = _require_string(output_root_value, "output_root")
    output_root = Path(_resolve_path(output_root_str, config_dir))

    stats_output_dir = config_payload.get("stats_output_dir")
    if stats_output_dir is None:
        stats_output_root = output_root / "stats"
    else:
        stats_output_root = Path(_resolve_path(_require_string(stats_output_dir, "stats_output_dir"), config_dir))

    update_output_dir = config_payload.get("update_output_dir")
    if update_output_dir is None:
        update_output_root = output_root / "update"
    else:
        update_output_root = Path(_resolve_path(_require_string(update_output_dir, "update_output_dir"), config_dir))

    global_defaults_raw = _ensure_mapping(config_payload.get("defaults"), "defaults")
    _unknown_keys(COMMON_OPTION_KEYS, global_defaults_raw, "defaults")

    stats_section = _ensure_mapping(config_payload.get("stats"), "stats")
    _unknown_keys(STATS_SECTION_KEYS, stats_section, "stats")
    stats_defaults_raw = _ensure_mapping(stats_section.get("defaults"), "stats.defaults")
    _unknown_keys(STATS_DEFAULT_KEYS, stats_defaults_raw, "stats.defaults")
    stats_jobs = _ensure_list(stats_section.get("jobs"), "stats.jobs")

    update_section = _ensure_mapping(config_payload.get("update"), "update")
    _unknown_keys(UPDATE_SECTION_KEYS, update_section, "update")
    update_defaults_raw = _ensure_mapping(update_section.get("defaults"), "update.defaults")
    _unknown_keys(UPDATE_DEFAULT_KEYS, update_defaults_raw, "update.defaults")
    update_jobs = _ensure_list(update_section.get("jobs"), "update.jobs")

    if not stats_jobs and not update_jobs:
        raise BatchConfigError("At least one job must be provided in stats.jobs or update.jobs.")

    jobs: list[dict[str, Any]] = []
    seen_outputs: dict[str, str] = {}
    idx = 1

    for job in stats_jobs:
        job_context = f"stats.jobs[{idx - 1}]"
        if not isinstance(job, dict):
            raise BatchConfigError(f"{job_context} must be a mapping.")
        _unknown_keys(STATS_OPTION_KEYS, job, job_context)

        merged = dict(STATS_DEFAULTS)
        merged.update(merge_settings(global_defaults_raw, stats_defaults_raw, job))
        merged = _validate_effective_common(merged, job_context)
        merged["metal_distance"] = _validate_number_in_range(
            "metal_distance", merged["metal_distance"], 0, 1, job_context
        )

        pdb_value = _require_string(merged.get("pdb"), f"{job_context}.pdb")
        merged["pdb"] = _resolve_pdb_value(pdb_value, config_dir)

        ligand = merged.get("ligand", "")
        if ligand is None:
            ligand = ""
        if not isinstance(ligand, str):
            raise BatchConfigError(f"{job_context}.ligand must be a string when provided.")
        merged["ligand"] = ligand

        if merged.get("debug_output") is not None:
            merged["debug_output"] = _normalize_optional_path(merged["debug_output"], config_dir)

        output_override = merged.get("output")
        if output_override is not None:
            output_path = _resolve_output_path(_require_string(output_override, f"{job_context}.output"), config_dir)
        else:
            pdb_stem = Path(str(merged["pdb"])).stem
            if ligand:
                output_path = str(stats_output_root / f"{pdb_stem}_{ligand}.json")
            else:
                output_path = str(stats_output_root / pdb_stem)

        if not ligand and Path(output_path).suffix:
            raise BatchConfigError(
                f"{job_context}: multi-ligand stats output must be a directory path without a suffix."
            )

        if merged["debug"] and merged.get("debug_output") and not ligand and Path(str(merged["debug_output"])).suffix:
            raise BatchConfigError(
                f"{job_context}: debug_output must be a directory for multi-ligand stats jobs."
            )

        merged["output"] = output_path
        output_key = str(Path(output_path))
        if output_key in seen_outputs:
            raise BatchConfigError(
                f"Output path collision detected: `{output_path}` is used by {seen_outputs[output_key]} and {job_context}."
            )
        seen_outputs[output_key] = job_context

        jobs.append(
            {
                "index": idx,
                "mode": "stats",
                "name": merged.get("name") or f"stats_{idx}",
                "inputs": {"ligand": ligand or None, "pdb": merged["pdb"]},
                "resolved_output": output_path,
                "options": merged,
            }
        )
        idx += 1

    for job in update_jobs:
        job_context = f"update.jobs[{idx - len(stats_jobs) - 1}]"
        if not isinstance(job, dict):
            raise BatchConfigError(f"{job_context} must be a mapping.")
        _unknown_keys(UPDATE_OPTION_KEYS, job, job_context)

        merged = dict(UPDATE_DEFAULTS)
        merged.update(merge_settings(global_defaults_raw, update_defaults_raw, job))
        merged = _validate_effective_common(merged, job_context)
        merged["cif"] = _validate_boolean("cif", merged["cif"], job_context)

        input_value = _require_string(merged.get("input"), f"{job_context}.input")
        merged["input"] = _resolve_path(input_value, config_dir)

        pdb_value = merged.get("pdb")
        if pdb_value is not None:
            if not isinstance(pdb_value, str) or not pdb_value.strip():
                raise BatchConfigError(f"{job_context}.pdb must be a non-empty string when provided.")
            merged["pdb"] = _resolve_pdb_value(pdb_value, config_dir)

        if merged.get("debug_output") is not None:
            merged["debug_output"] = _normalize_optional_path(merged["debug_output"], config_dir)

        output_override = merged.get("output")
        if output_override is not None:
            output_path = _resolve_output_path(_require_string(output_override, f"{job_context}.output"), config_dir)
        else:
            output_path = str(update_output_root / f"{Path(merged['input']).stem}.cif")

        merged["output"] = output_path
        output_key = str(Path(output_path))
        if output_key in seen_outputs:
            raise BatchConfigError(
                f"Output path collision detected: `{output_path}` is used by {seen_outputs[output_key]} and {job_context}."
            )
        seen_outputs[output_key] = job_context

        jobs.append(
            {
                "index": idx,
                "mode": "update",
                "name": merged.get("name") or f"update_{idx}",
                "inputs": {"input": merged["input"], "pdb": merged.get("pdb"), "cif": merged["cif"]},
                "resolved_output": output_path,
                "options": merged,
            }
        )
        idx += 1

    return output_root, jobs


def handle_batch(args) -> None:
    config_path = Path(args.config).resolve()
    started_at = _iso_now()

    try:
        payload = _load_yaml(config_path)
        output_root, jobs = _plan_jobs(payload, config_path)
    except BatchConfigError as exc:
        Logger().error(str(exc))
        raise SystemExit(2) from exc

    records: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    if args.dry_run:
        for job in jobs:
            now = _iso_now()
            records.append(
                {
                    "index": job["index"],
                    "mode": job["mode"],
                    "name": str(job["name"]),
                    "inputs": job["inputs"],
                    "resolved_output": job["resolved_output"],
                    "status": "dry_run",
                    "reason": None,
                    "started_at": now,
                    "finished_at": now,
                    "duration_sec": 0.0,
                }
            )
        success_count = len(jobs)
        failure_count = 0
        overall_status = "dry_run"
    else:
        for job in jobs:
            job_started = datetime.now(timezone.utc)
            status_path = status_path_for_output(job["resolved_output"])
            status_file = Path(status_path)
            if status_file.exists():
                status_file.unlink()

            if job["mode"] == "stats":
                run_args = _build_stats_args(job["options"])
                handle_stats(run_args)
            else:
                run_args = _build_update_args(job["options"])
                handle_update(run_args)

            job_finished = datetime.now(timezone.utc)
            status_payload = read_status_for_output(job["resolved_output"])
            status = status_payload.get("status", "Failure")
            reason = status_payload.get("Reason")
            is_success = status == "Success"

            if is_success:
                success_count += 1
                job_status = "success"
            else:
                failure_count += 1
                job_status = "failure"

            records.append(
                {
                    "index": job["index"],
                    "mode": job["mode"],
                    "name": str(job["name"]),
                    "inputs": job["inputs"],
                    "resolved_output": job["resolved_output"],
                    "status": job_status,
                    "reason": reason,
                    "started_at": job_started.isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "finished_at": job_finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "duration_sec": round((job_finished - job_started).total_seconds(), 3),
                }
            )

        overall_status = "success" if failure_count == 0 else "partial_failure"

    finished_at = _iso_now()
    report_payload = {
        "meta": {
            "config": str(config_path),
            "started_at": started_at,
            "finished_at": finished_at,
            "dry_run": bool(args.dry_run),
            "total": len(jobs),
            "success_count": success_count,
            "failure_count": failure_count,
        },
        "jobs": records,
        "status": overall_status,
    }

    _write_batch_reports(output_root, report_payload)

    if args.dry_run:
        return
    if failure_count > 0:
        raise SystemExit(1)
