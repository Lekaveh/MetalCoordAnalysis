import json
import os
from pathlib import Path
from typing import Optional

from metalCoord.logging import Logger
from metalCoord.config import Config


def configure_statistics(args, config: Config) -> Config:
    if args.min_size > args.max_size:
        raise ValueError("Minimum sample size must be less or equal than maximum sample size.")

    config.ideal_angles = getattr(args, "ideal_angles", False)
    config.distance_threshold = args.dist
    config.procrustes_threshold = args.threshold
    config.min_sample_size = args.min_size
    config.simple = getattr(args, "simple", False)
    config.save = getattr(args, "save", False)
    config.use_pdb = getattr(args, "use_pdb", False)
    config.output_folder = os.path.abspath(os.path.dirname(args.output))
    config.output_file = os.path.basename(args.output)
    config.max_coordination_number = args.coordination
    config.max_sample_size = args.max_size
    return config


def write_status(
    status: str,
    config: Config,
    reason: Optional[str] = None,
    ensure_dir: bool = False,
) -> None:
    if ensure_dir:
        Path(config.output_folder).mkdir(exist_ok=True, parents=True)
    status_path = os.path.join(config.output_folder, config.output_file + ".status.json")
    payload = {"status": status}
    if reason:
        payload["Reason"] = reason
    with open(status_path, 'w', encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4, separators=(',', ': '))


def log_exception(exc: Exception) -> None:
    Logger().error(f"{str(exc)}")
