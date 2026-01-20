from metalCoord.service.info import process_coordinations

from metalCoord.cli.commands.common import log_exception


def handle_coord(args) -> None:
    try:
        process_coordinations(args.number, args.metal, args.output, getattr(args, "cod", False))
    except Exception as exc:
        log_exception(exc)
