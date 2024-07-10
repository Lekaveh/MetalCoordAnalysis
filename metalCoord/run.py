import argparse
import os
import metalCoord
from metalCoord.logging import Logger
from metalCoord.services import update_cif, get_stats, get_coordinations
from metalCoord.config import Config


class Range:
    """
    Represents a range of values between a start and end point.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return f"range ({self.start}, {self.end})"

    def __str__(self):
        return f"range ({self.start}, {self.end})"


def check_positive(value: str) -> int:
    """
    Check if a value is a positive integer.

    Args:
        value (str): The value to check.

    Returns:
        int: The integer value of the input.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value")
    return ivalue


def create_parser():
    """
    Create the argument parser for the MetalCoord command-line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(
        prog='metalCoord', description='MetalCoord: Metal coordination analysis.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + metalCoord.__version__)

    # Define the subparsers for the two apps
    subparsers = parser.add_subparsers(dest='command')

    # App1
    update_parser = subparsers.add_parser('update', help='Update a cif file.')
    update_parser.add_argument(
        '-i', '--input', type=str, required=True, help='CIF file.', metavar='<INPUT CIF FILE>')
    update_parser.add_argument('-o', '--output', type=str, required=True,
                               help='Output cif file.', metavar='<OUTPUT CIF FILE>')
    update_parser.add_argument('-p', '--pdb', type=str, required=False,
                               help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    update_parser.add_argument('-d', '--dist', type=float, required=False, help='Distance threshold.',
                               metavar='<DISTANCE THRESHOLD>', default=0.2, choices=[Range(0, 1)])
    update_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.',
                               metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    update_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.',
                               metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)
    update_parser.add_argument('--ideal-angles', required=False,
                               help='Provide only ideal angles', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('-s', '--simple', required=False,
                              help='Simple distance based filtering', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('--save', required=False,
                              help='Save COD files used in statistics', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('--use-pdb', required=False,
                              help='Use COD structures based on pdb coordinates', default=argparse.SUPPRESS,  action='store_true')

    # App2
    stats_parser = subparsers.add_parser(
        'stats', help='Distance and angle statistics.')
    stats_parser.add_argument('-l', '--ligand', type=str,
                              required=True, help='Ligand code.', metavar='<LIGAND CODE>')
    stats_parser.add_argument('-p', '--pdb', type=str, required=True,
                              help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    stats_parser.add_argument('-o', '--output', type=str, required=True,
                              help='Output json file.', metavar='<OUTPUT JSON FILE>')
    stats_parser.add_argument('-d', '--dist', type=float, required=False, help='Distance threshold.',
                              metavar='<DISTANCE THRESHOLD>', default=0.2, choices=[Range(0, 1)])
    stats_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.',
                              metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    stats_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.',
                              metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)
    stats_parser.add_argument('--ideal-angles', required=False,
                              help='Provide only ideal angles', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('-s', '--simple', required=False,
                              help='Simple distance based filtering', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('--save', required=False,
                              help='Save COD files used in statistics', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('--use-pdb', required=False,
                              help='Use COD structures based on pdb coordinates', default=argparse.SUPPRESS,  action='store_true')


    # App3
    coordination_parser = subparsers.add_parser(
        'coord', help='List of coordinations.')
    coordination_parser.add_argument('-n', '--number', type=int, required=False,
                                     help='Coordination number.', metavar='<COORDINATION NUMBER>')
    return parser


def main_func():
    """
    The main function of the MetalCoord command-line interface.

    It initializes the logger, parses the command-line arguments, and executes the appropriate command.

    Raises:
        argparse.ArgumentError: If the command-line arguments are invalid.
    """
    try:
        Logger().add_handler()
        Logger().info(f"Logging started. Logging level: {Logger().logger.level}")

        parser = create_parser()
        args = parser.parse_args()


        if args.command == 'update' or args.command == 'stats':
            Config().ideal_angles = args.ideal_angles if "ideal_angles" in args else False
            Config().distance_threshold = args.dist
            Config().procrustes_threshold = args.threshold
            Config().min_sample_size = args.min_size
            Config().simple = args.simple if "simple" in args else False
            Config().save = args.save if "save" in args else False
            Config().use_pdb = args.use_pdb if "use_pdb" in args else False
            Config().output_folder = os.path.dirname(args.output)


        if args.command == 'update':
            update_cif(args.output, args.input, args.pdb)

        elif args.command == 'stats':
            get_stats(args.ligand, args.pdb, args.output)
        
        elif args.command == 'coord':
            print(f"List of coordinations: {get_coordinations(args.number)}")
        else:
            parser.print_help()
    except ValueError as e:
        Logger().error(f"{repr(e)}")
    except FileNotFoundError as e:
        Logger().error(f"{repr(e)}")
    except PermissionError as e:
        Logger().error(f"{repr(e)}")
    except Exception as e:
        Logger().error(f"{repr(e)}")


if __name__ == '__main__':
    main_func()
