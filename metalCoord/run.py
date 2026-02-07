from metalCoord.cli.commands.coord import handle_coord
from metalCoord.cli.commands.pdb import handle_pdb
from metalCoord.cli.commands.stats import handle_stats
from metalCoord.cli.commands.update import handle_update
from metalCoord.cli.parser import create_parser
from metalCoord.logging import Logger


def main_func():
    """
    The main function of the MetalCoord command-line interface.

    It initializes the logger, parses the command-line arguments, and executes the appropriate command.

    Raises:
        argparse.ArgumentError: If the command-line arguments are invalid.
    """

    parser = create_parser()
    args = parser.parse_args()
    Logger().add_handler(True, not args.no_progress)
    if getattr(args, "debug", False):
        Logger().enable_capture(True, reset=True)
    Logger().info(
        f"Logging started. Logging level: {Logger().logger.level}")

    handlers = {
        'update': handle_update,
        'stats': handle_stats,
        'coord': handle_coord,
        'pdb': handle_pdb,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main_func()
