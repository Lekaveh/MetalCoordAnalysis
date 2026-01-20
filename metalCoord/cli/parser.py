import argparse

import metalCoord


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


def check_positive_more_than_two(value: str) -> int:
    """
    Check if the given value is a positive integer greater than 1.

    Args:
        value (str): The value to be checked.

    Returns:
        int: The converted integer value if it is valid.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid positive integer greater than 1.
    """

    ivalue = int(value)
    if ivalue <= 1:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value. It should be more than 1")
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

    parser.add_argument('--no-progress', required=False,
                        help='Do not show progress bar.', action='store_true')

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
                               metavar='<DISTANCE THRESHOLD>', default=0.5, choices=[Range(0, 1)])
    update_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.',
                               metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    update_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.',
                               metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)
    update_parser.add_argument('-x', '--max_size', required=False, help='Maximum sample size for statistics.',
                               metavar='<MAXIMUM SAMPLE SIZE>', default=2000, type=check_positive)
    update_parser.add_argument('--ideal-angles', required=False,
                               help='Provide only ideal angles', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('-s', '--simple', required=False,
                               help='Simple distance based filtering', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('--save', required=False,
                               help='Save COD files used in statistics', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('--use-pdb', required=False,
                               help='Use COD structures based on pdb coordinates', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('-c', '--coordination', type=check_positive_more_than_two, required=False,
                                     help='Maximum coordination number.', metavar='<MAXIMUM COORDINATION NUMBER>', default=1000)
    update_parser.add_argument('--cif', required=False,
                               help='Read coordinates from mmCIF file', default=argparse.SUPPRESS,  action='store_true')
    update_parser.add_argument('--cl', required=False,
                               help='Predefined class/coordination', metavar='<CLASS>', type=str)

    # App2
    stats_parser = subparsers.add_parser(
        'stats', help='Distance and angle statistics.')
    stats_parser.add_argument('-l', '--ligand', type=str,
                              required=False, help='Ligand code.', metavar='<LIGAND CODE>', default="")
    stats_parser.add_argument('-p', '--pdb', type=str, required=True,
                              help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    stats_parser.add_argument('-o', '--output', type=str, required=True,
                              help='Output json file.', metavar='<OUTPUT JSON FILE>')
    stats_parser.add_argument('-d', '--dist', type=float, required=False, help='Distance threshold.',
                              metavar='<DISTANCE THRESHOLD>', default=0.5, choices=[Range(0, 1)])
    stats_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.',
                              metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    stats_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.',
                              metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)
    stats_parser.add_argument('-x', '--max_size', required=False, help='Maximum sample size for statistics.',
                               metavar='<MAXIMUM SAMPLE SIZE>', default=2000, type=check_positive)
    stats_parser.add_argument('--ideal-angles', required=False,
                              help='Provide only ideal angles', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('-s', '--simple', required=False,
                              help='Simple distance based filtering', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('--save', required=False,
                              help='Save COD files used in statistics', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('--use-pdb', required=False,
                              help='Use COD structures based on pdb coordinates', default=argparse.SUPPRESS,  action='store_true')
    stats_parser.add_argument('-c', '--coordination', type=check_positive_more_than_two, required=False,
                              help='Maximum coordination number.', metavar='<MAXIMUM COORDINATION NUMBER>', default=1000)
    stats_parser.add_argument('--cl', required=False,
                               help='Predefined class/coordination', metavar='<CLASS>', type=str)
    stats_parser.add_argument('--metal_distance', type=float, required=False, help='Metal Metal distance threshold.',
                              metavar='<METAL DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])

    # App3
    coordination_parser = subparsers.add_parser(
        'coord', help='List of coordinations.')
    coordination_parser.add_argument('-n', '--number', type=int, required=False,
                                     help='Coordination number.', metavar='<COORDINATION NUMBER>')
    coordination_parser.add_argument('-m', '--metal', type=str, required=False)
    coordination_parser.add_argument('-o', '--output', type=str, required=False,
                                     help='Output json file.', metavar='<OUTPUT JSON FILE>')
    coordination_parser.add_argument('--cod', required=False,
                              help='Include IDs of the COD structures.', default=argparse.SUPPRESS,  action='store_true')

    # App4
    pdb_parser = subparsers.add_parser(
        'pdb', help='Get list of PDBs containing the ligand.')
    pdb_parser.add_argument('-l', '--ligand', type=str,
                            required=True, help='Ligand code.', metavar='<LIGAND CODE>')
    pdb_parser.add_argument('-o', '--output', type=str, required=False,
                            help='Output json file.', metavar='<OUTPUT JSON FILE>')
    return parser
