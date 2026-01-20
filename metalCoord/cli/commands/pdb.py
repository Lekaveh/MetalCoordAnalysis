import os

from metalCoord.service.info import process_pdbs_list

from metalCoord.cli.commands.common import log_exception, write_status
from metalCoord.config import Config


def handle_pdb(args, config: Config) -> None:
    try:
        if args.output:
            config.output_folder = os.path.abspath(os.path.dirname(args.output))
            config.output_file = os.path.basename(args.output)
        process_pdbs_list(args.ligand, args.output)
        write_status("Success", config)
    except Exception as exc:
        log_exception(exc)
