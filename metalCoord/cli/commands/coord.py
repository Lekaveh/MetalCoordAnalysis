from metalCoord.service.info import process_coordinations

from metalCoord.cli.commands.common import log_exception
from metalCoord.config import Config


def handle_coord(args, config: Config) -> None:
    try:
        process_coordinations(args.number, args.metal, args.output, getattr(args, "cod", False))
    except Exception as exc:
        log_exception(exc)
