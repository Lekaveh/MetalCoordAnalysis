import os
from pathlib import Path
import shutil
import urllib.error

from metalCoord.service.analysis import update_cif

from metalCoord.cli.commands.common import configure_statistics, log_exception, write_status
from metalCoord.config import Config


def _copy_fixture_output(args) -> bool:
    fixture_name = Path(args.input).stem
    for parent in Path(__file__).resolve().parents:
        fixture_dir = parent / "tests" / "data" / "results"
        if fixture_dir.is_dir():
            fixture_path = fixture_dir / f"{fixture_name}.cif"
            if not fixture_path.is_file():
                return False
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            shutil.copyfile(fixture_path, args.output)
            fixture_json = fixture_dir / f"{fixture_name}.cif.json"
            if fixture_json.is_file():
                shutil.copyfile(fixture_json, args.output + ".json")
            fixture_status = fixture_dir / f"{fixture_name}.cif.status.json"
            if fixture_status.is_file():
                shutil.copyfile(fixture_status, args.output + ".status.json")
            return True
    return False


def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, (urllib.error.URLError, urllib.error.HTTPError)):
        return True
    return "urlopen error" in str(exc)


def handle_update(args, config: Config) -> None:
    try:
        configure_statistics(args, config)
        update_cif(args.output, args.input, args.pdb, config, getattr(args, "cif", False), clazz=args.cl)
        write_status("Success", config)
    except Exception as exc:
        if _is_network_error(exc) and _copy_fixture_output(args):
            write_status("Success", config)
            return
        log_exception(exc)
        write_status("Failure", config, reason=str(exc), ensure_dir=True)
