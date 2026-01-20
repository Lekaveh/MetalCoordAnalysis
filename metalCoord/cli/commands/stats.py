from metalCoord.service.analysis import get_stats

from metalCoord.cli.commands.common import configure_statistics, log_exception, write_status
from metalCoord.config import Config


def handle_stats(args, config: Config) -> None:
    try:
        configure_statistics(args, config)
        config.metal_distance_threshold = args.metal_distance
        get_stats(args.ligand, args.pdb, args.output, config, clazz=args.cl)
        write_status("Success", config)
    except Exception as exc:
        log_exception(exc)
        write_status("Failure", config, reason=str(exc), ensure_dir=True)
