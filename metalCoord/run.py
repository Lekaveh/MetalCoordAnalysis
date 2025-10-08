"""Entry point for the MetalCoord command-line interface."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from metalCoord.cli import (
    AnalysisCommandArgs,
    CommandResult,
    CoordinationArgs,
    PdbArgs,
    StatsArgs,
    UpdateArgs,
    parse_cli_args,
)
from metalCoord.config import Config
from metalCoord.logging import Logger


def _configure_logging(no_progress: bool) -> None:
    logger = Logger()
    logger.add_handler(enable=True, progress_bars=not no_progress)
    logger.info(f"Logging started. Logging level: {logger.logger.level}")


def _status_path(config: Config) -> Path | None:
    if not config.output_file:
        return None
    folder = Path(config.output_folder or os.getcwd())
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{config.output_file}.status.json"


def _write_status(status: dict[str, Any]) -> None:
    config = Config()
    path = _status_path(config)
    if not path:
        return
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(status, json_file, indent=4, separators=(",", ": "))


def _apply_analysis_config(args: AnalysisCommandArgs) -> None:
    config = Config()
    args.apply_to_config(config)


def _run_update(args: UpdateArgs) -> None:
    from metalCoord.service.analysis import update_cif

    _apply_analysis_config(args)
    update_cif(args.output, args.input, args.pdb, args.cif, clazz=args.clazz)


def _run_stats(args: StatsArgs) -> None:
    from metalCoord.service.analysis import get_stats

    _apply_analysis_config(args)
    get_stats(args.ligand, args.pdb, args.output, clazz=args.clazz)


def _run_coord(args: CoordinationArgs) -> None:
    from metalCoord.service.info import process_coordinations

    process_coordinations(args.number, args.metal, args.output, args.cod)


def _run_pdb(args: PdbArgs) -> None:
    from metalCoord.service.info import process_pdbs_list

    process_pdbs_list(args.ligand, args.output)


def _dispatch_command(command: CommandResult[Any]) -> None:
    if isinstance(command.args, UpdateArgs):
        _run_update(command.args)
    elif isinstance(command.args, StatsArgs):
        _run_stats(command.args)
    elif isinstance(command.args, CoordinationArgs):
        _run_coord(command.args)
    elif isinstance(command.args, PdbArgs):
        _run_pdb(command.args)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown command: {command.command}")


def main_func(argv: list[str] | None = None) -> None:
    command: CommandResult[Any] | None = None
    try:
        command = parse_cli_args(argv)
        _configure_logging(command.no_progress)
        _dispatch_command(command)
        if isinstance(command.args, (UpdateArgs, StatsArgs)):
            _write_status({"status": "Success"})
    except Exception as exc:  # pragma: no cover - entry point safety
        Logger().error(str(exc))
        if command and isinstance(command.args, (UpdateArgs, StatsArgs)):
            _write_status({"status": "Failure", "Reason": str(exc)})


if __name__ == "__main__":  # pragma: no cover
    main_func()
