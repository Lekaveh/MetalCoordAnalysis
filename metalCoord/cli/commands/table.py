from metalCoord.cli.commands.common import log_exception
from metalCoord.service.info import process_table


def handle_table(args) -> None:
    try:
        process_table(args.output_folder, args.format)
    except Exception as exc:
        log_exception(exc)
