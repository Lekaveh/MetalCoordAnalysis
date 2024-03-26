import argparse
import tensorflow as tf
from metalCoord.logging import Logger
from metalCoord.services import update_cif, get_stats
from metalCoord.config import Config

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
    
    def __repr__(self):
        return f"range ({self.start}, {self.end})"
    
    def __str__(self):
        return f"range ({self.start}, {self.end})"

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def create_parser():
    parser = argparse.ArgumentParser(prog='metalCoord', description='MetalCoord: Metal coordination analysis.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.13')

    # Define the subparsers for the two apps    
    subparsers = parser.add_subparsers(dest='command')

    # App1
    update_parser = subparsers.add_parser('update', help='Update a cif file.')
    update_parser.add_argument('-i', '--input', type=str, required=True, help='CIF file.', metavar='<INPUT CIF FILE>')
    update_parser.add_argument('-o', '--output', type=str, required=True, help='Output cif file.', metavar='<OUTPUT CIF FILE>')
    update_parser.add_argument('-p', '--pdb', type=str, required=False, help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    update_parser.add_argument('-d', '--dist', type=float, required=False, help='Distance threshold.', metavar='<DISTANCE THRESHOLD>', default=0.2, choices=[Range(0, 1)])
    update_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.', metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    update_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.', metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)

    # App2
    stats_parser = subparsers.add_parser('stats', help='Distance and angle statistics.')
    stats_parser.add_argument('-l', '--ligand', type=str, required=True, help='Ligand code.', metavar='<LIGAND CODE>')
    stats_parser.add_argument('-p', '--pdb', type=str, required=True, help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    stats_parser.add_argument('-o', '--output', type=str, required=True, help='Output json file.', metavar='<OUTPUT JSON FILE>')
    stats_parser.add_argument('-d', '--dist', type=float, required=False, help='Distance threshold.', metavar='<DISTANCE THRESHOLD>', default=0.2, choices=[Range(0, 1)])
    stats_parser.add_argument('-t', '--threshold', type=float, required=False, help='Procrustes distance threshold.', metavar='<PROCRUSTES DISTANCE THRESHOLD>', default=0.3, choices=[Range(0, 1)])
    stats_parser.add_argument('-m', '--min_size', required=False, help='Minimum sample size for statistics.', metavar='<MINIMUM SAMPLE SIZE>', default=30, type=check_positive)



    return parser


def main_func():
    try:
        Logger().addHandler()
        Logger().info(f"Logging started. Logging level: {Logger().logger.level}")
        tf.config.set_visible_devices([], 'GPU')
        parser = create_parser()
        args = parser.parse_args()
        
        Config().distance_threshold = args.dist
        Config().procrustes_threshold = args.threshold
        Config().min_sample_size = args.min_size
        if args.command == 'update':
            update_cif(args.output, args.input, args.pdb)

        elif args.command == 'stats':
            get_stats(args.ligand, args.pdb, args.output)
        else:
            parser.print_help()
    except Exception as e:
        Logger().error(f"{repr(e)}")

if __name__ == '__main__':
    main_func()    