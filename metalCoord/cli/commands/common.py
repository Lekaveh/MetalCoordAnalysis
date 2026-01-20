import json
import os
from pathlib import Path
from typing import Optional

from metalCoord.config import Config
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


def write_status(status: str, reason: Optional[str] = None, ensure_dir: bool = False) -> None:
    if ensure_dir:
        Path(Config().output_folder).mkdir(exist_ok=True, parents=True)
    status_path = os.path.join(Config().output_folder, Config().output_file + ".status.json")
    payload = {"status": status}
    if reason:
        payload["Reason"] = reason
    with open(status_path, 'w', encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4, separators=(',', ': '))


def log_exception(exc: Exception) -> None:
    Logger().error(f"{str(exc)}")
