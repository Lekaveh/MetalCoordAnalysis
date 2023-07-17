from gemmi import cif
import json
from analysis.stats import find_classes, get_structures
import os
import gemmi
import os
from pathlib import Path


def decompose(values, n):
    result = []
    for i in range(0, len(values)//n):
        result.append(values[i*n:i*n + n])
    return result
def pack(values):
    return [list(x) for x in zip(*values)]  

mons = json.load(open("./data/mons.json"))


def get_stats(results, metal_name, ligand_name):
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
                    if ligand["ligand"] == ligand_name:
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
        loop = block.find_loop ('_chem_comp_bond.value_dist').get_loop()
        rows = decompose(loop.values, len(loop.tags))
        if name in mons and contains_metal(block):
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
                distance, std, cl, procrustes, coordination = get_stats(results, metal_name, ligand_name)
                if coordination > 0:
                    row[4] = row[6] = str(round(distance, 3))
                    row[5] = row[7] = str(round(std, 3))
            
            loop.set_all_values(pack(rows))
            out = os.path.join(output_path, folder)
            Path(out).mkdir(exist_ok=True, parents=True)
            doc.write_file(os.path.join(out, f"{name}.cif"))
    except Exception as e:
        print(e)
        print(path)

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