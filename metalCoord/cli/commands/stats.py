from metalCoord.service.analysis import get_stats

from metalCoord.cli.commands.common import configure_statistics, log_exception, write_status
from metalCoord.config import Config


def handle_stats(args) -> None:
    try:
        configure_statistics(args)
        Config().metal_distance_threshold = args.metal_distance
        get_stats(args.ligand, args.pdb, args.output, clazz=args.cl)
        write_status("Success")
    except Exception as exc:
        log_exception(exc)
        write_status("Failure", reason=str(exc), ensure_dir=True)
