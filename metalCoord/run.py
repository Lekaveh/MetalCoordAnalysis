import argparse
import tensorflow as tf
from metalCoord.logging import Logger
from metalCoord.services import update_cif, get_stats

def create_parser():
    parser = argparse.ArgumentParser(prog='metalCoord', description='MetalCoord: Metal coordination analysis.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.5')

    # Define the subparsers for the two apps    
    subparsers = parser.add_subparsers(dest='command')

    # App1
    update_parser = subparsers.add_parser('update', help='Update a cif file.')
    update_parser.add_argument('-i', '--input', type=str, required=True, help='CIF file.', metavar='<INPUT CIF FILE>')
    update_parser.add_argument('-o', '--output', type=str, required=True, help='Output cif file.', metavar='<OUTPUT CIF FILE>')
    update_parser.add_argument('-p', '--pdb', type=str, required=False, help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')

    # App2
    stats_parser = subparsers.add_parser('stats', help='Distance and angle statistics.')
    stats_parser.add_argument('-l', '--ligand', type=str, required=True, help='Ligand code.', metavar='<LIGAND CODE>')
    stats_parser.add_argument('-p', '--pdb', type=str, required=True, help='PDB code or pdb file.', metavar='<PDB CODE|PDB FILE>')
    stats_parser.add_argument('-o', '--output', type=str, required=True, help='Output json file.', metavar='<OUTPUT JSON FILE>')

    return parser


def main_func():
    try:
        Logger().addHandler()
        Logger().info(f"Logging started. Logging level: {Logger().logger.level}")
        tf.config.set_visible_devices([], 'GPU')
        parser = create_parser()
        args = parser.parse_args()
        

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