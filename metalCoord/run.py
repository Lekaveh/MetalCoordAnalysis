from gemmi import cif
import json
from metalCoord.analysis.stats import find_classes, get_structures
import os
import gemmi
import sys
from pathlib import Path
import argparse
import json



def decompose(values, n):
    result = []
    for i in range(0, len(values)//n):
        result.append(values[i*n:i*n + n])
    return result
def pack(values):
    return [list(x) for x in zip(*values)]  

d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json")))


def get_distance(results, metal_name, ligand_name):
    coordination = 0
    cl = ""
    procrustes = 1
    distance = -1
    std = -1

    for metal in results:
        if metal["metal"] == metal_name:
            for clazz in metal["ligands"]:
                if clazz["coordination"] < coordination:
                        continue
                for ligand in clazz["base"]:
                    if ligand["ligand"]["name"]  == ligand_name:
                        if clazz["coordination"] == coordination and  clazz["procrustes"] < procrustes:
                            coordination =  clazz["coordination"]
                            procrustes   =  clazz["procrustes"]
                            distance = ligand["distance"]
                            std = ligand["std"]
                            cl = clazz["class"]

                        if clazz["coordination"] > coordination:
                            coordination =  clazz["coordination"]
                            procrustes   =  clazz["procrustes"]
                            distance = ligand["distance"]
                            std = ligand["std"]
                            cl = clazz["class"]
                        
                        break
    return distance, std, cl, procrustes, coordination

def get_angles(results, metal_name, ligand_name1, ligand_name2):
    coordination = 0
    cl = ""
    procrustes = 1
    angle = -1
    std = -1

    for metal in results:
        if metal["metal"] == metal_name:
            for clazz in metal["ligands"]:
                if clazz["coordination"] < coordination:
                        continue
                for ligand in clazz["angles"]:
                    if ((ligand["ligand1"]["name"] == ligand_name1) and (ligand["ligand2"]["name"]  == ligand_name2)) or ((ligand["ligand2"]["name"]  == ligand_name1) and (ligand["ligand1"]["name"]  == ligand_name2)):
                        if clazz["coordination"] == coordination and  clazz["procrustes"] < procrustes:
                            coordination =  clazz["coordination"]
                            procrustes   =  clazz["procrustes"]
                            angle = ligand["angle"]
                            std = ligand["std"]
                            cl = clazz["class"]

                        if clazz["coordination"] > coordination:
                            coordination =  clazz["coordination"]
                            procrustes   =  clazz["procrustes"]
                            angle = ligand["angle"]
                            std = ligand["std"]
                            cl = clazz["class"]
                        
                        break
    return angle, std, cl, procrustes, coordination


def differ(path):
    try:
        folder, name = os.path.split(path)
        name = name[:-4]
        folder = os.path.split(folder)[1]
        doc = cif.read_file(path)
        block = doc.find_block(f"comp_{name}")
        if name in mons:
            el_name = get_element_name_dict(block)
            pdb = mons[name][0][0]
            structures = get_structures(name, pdb)
            return [name, len(structures) > 0, contains_metal(block)]
    except Exception as e:
        print(e)
        print(path)
    
    return [name, None, None]


def contains_metal(block):
    loop = block.find_loop ('_chem_comp_atom.comp_id').get_loop()
    rows = decompose(loop.values, len(loop.tags))
    for row in rows:
        if gemmi.Element(row[2]).is_metal:
            return True
    return False

def get_element_name_dict(block):
    result = dict()
    loop = block.find_loop ('_chem_comp_atom.comp_id').get_loop()
    rows = decompose(loop.values, len(loop.tags))
    for row in rows:
        result[row[1]] =  row[2]
    return result


def adjust(output_path, path):
    try:
        folder, name = os.path.split(path)
        name = name[:-4]
        folder = os.path.split(folder)[1]
        doc = cif.read_file(path)
        block = doc.find_block(f"comp_{name}")
        
        
        if name in mons and contains_metal(block):
            loop = block.find_loop ('_chem_comp_bond.value_dist').get_loop()
            rows = decompose(loop.values, len(loop.tags))
            el_name = get_element_name_dict(block)
            pdb = mons[name][0][0]
            results = find_classes(name, pdb)
            for row in rows:
                
                metal_name = row[1]
                ligand_name = row[2]
                if not gemmi.Element(el_name[row[2]]).is_metal and not gemmi.Element(el_name[row[1]]).is_metal:
                    continue

                if gemmi.Element(row[2]).is_metal:
                    metal_name = row[2]
                    ligand_name = row[1]


                distance, std, cl, procrustes, coordination = get_distance(results, metal_name, ligand_name)
                if coordination > 0:
                    row[4] = row[6] = str(round(distance, 3))
                    row[5] = row[7] = str(round(std, 3))
            
            loop.set_all_values(pack(rows))



            loop = block.find_loop ('_chem_comp_angle.value_angle').get_loop()
            rows = decompose(loop.values, len(loop.tags))
            if name in mons and contains_metal(block):
                el_name = get_element_name_dict(block)
                pdb = mons[name][0][0]
                results = find_classes(name, pdb)
                for row in rows:
                    
                    ligand1_name = row[1]
                    metal_name = row[2]
                    ligand2_name = row[3]
                    if not gemmi.Element(el_name[row[2]]).is_metal:
                        continue

                    angle, std, cl, procrustes, coordination = get_angles(results, metal_name, ligand1_name, ligand2_name)
                    if coordination > 0:
                        row[4]  = str(round(angle, 3))
                        row[5]  = str(round(std, 3))
                
            loop.set_all_values(pack(rows))

            Path(os.path.split(output_path)[0]).mkdir(exist_ok=True, parents=True)
            doc.write_file(output_path)
    except Exception as e:
        print(e)
        print(path)


def create_parser():
    parser = argparse.ArgumentParser()

    # Define the subparsers for the two apps
    subparsers = parser.add_subparsers(dest='command')

    # App1
    update_parser = subparsers.add_parser('update', help='Update a cif file.')
    update_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input CIF file.')
    update_parser.add_argument('-o', '--output', type=str, required=True, help='Updated CIF file path.')

    # App2
    stats_parser = subparsers.add_parser('stats', help='Distance statistics.')
    stats_parser.add_argument('-l', '--ligand', type=str, required=True, help='Ligand namr.')
    stats_parser.add_argument('-p', '--pdb', type=str, required=True, help='PDB name.')
    stats_parser.add_argument('-o', '--output', type=str, required=True, help='Output file path.')

    return parser


def main_func():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == 'update':
        adjust(args.output, args.input)

    elif args.command == 'stats':
        results = find_classes(args.ligand, args.pdb)
        with open(args.output, 'w') as json_file:
            json.dump(results, json_file, 
                                indent=4,  
                                separators=(',',': '))

if __name__ == '__main__':
    main_func()    