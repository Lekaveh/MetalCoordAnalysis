

import argparse
import json
from analysis.stats import find_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', required=True, help='ligand name')
    parser.add_argument('--pdb', required=True, help='pdb name')
    parser.add_argument('--output', required=True, help='output file path')

    args = parser.parse_args()

    ligand = args.ligand
    pdb_name = args.pdb
    output = args.output


    results = find_classes(ligand, pdb_name)
    with open(output, 'w') as json_file:
        json.dump(results, json_file, 
                            indent=4,  
                            separators=(',',': '))
