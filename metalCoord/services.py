import json
import os
from pathlib import Path
import sys
import gemmi
from metalCoord.analysis.stats import find_classes
from metalCoord.logging import Logger



_angle_category = "_chem_comp_angle"
_bond_category = "_chem_comp_bond"
_atom_category = "_chem_comp_atom"
_comp_category = "_chem_comp"
_comp_id = "comp_id"
_atom_id = "atom_id"
_type_symbol = "type_symbol"
_energy = "type_energy"
_atom_id_1 = "atom_id_1"
_atom_id_2 = "atom_id_2"
_atom_id_3 = "atom_id_3"
_value_dist_nucleus = "value_dist_nucleus"
_value_dist_nucleus_esd = "value_dist_nucleus_esd"
_value_dist = "value_dist"
_value_dist_esd = "value_dist_esd"
_value_angle = "value_angle"
_value_angle_esd = "value_angle_esd"
_id = "id"
_name = "name"
_group = "group"
_number_atoms_all = "number_atoms_all"
_number_atoms_nh = "number_atoms_nh"
_desc_level = "desc_level"
_three_letter_code = "three_letter_code"


def decompose(values, n):
    result = []
    for i in range(0, len(values)//n):
        result.append(values[i*n:i*n + n])
    return result
def pack(values):
    return [list(x) for x in zip(*values)]  

d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json")))


def get_distance(clazz, ligand_name):
    coordination = 0
    cl = ""
    procrustes = 1
    distance = -1
    std = -1


    if clazz is None:
        return distance, std, cl, procrustes, coordination

    for ligand in clazz["base"]:
        if ligand["ligand"]["name"]  == ligand_name:
            coordination =  clazz["coordination"]
            procrustes   =  float(clazz["procrustes"])
            distance = ligand["distance"]
            std = ligand["std"]
            cl = clazz["class"]
            break
                        
    return distance, std, cl, procrustes, coordination

def get_angles(clazz, ligand_name1, ligand_name2):
    coordination = 0
    cl = ""
    procrustes = 1
    angle = -1
    std = -1

    if clazz is None:
        return angle, std, cl, procrustes, coordination

    for ligand in clazz["angles"]:
        if ((ligand["ligand1"]["name"] == ligand_name1) and (ligand["ligand2"]["name"]  == ligand_name2)) or ((ligand["ligand2"]["name"]  == ligand_name1) and (ligand["ligand1"]["name"]  == ligand_name2)):
            coordination =  clazz["coordination"]
            procrustes   =  float(clazz["procrustes"])
            angle = ligand["angle"]
            std = ligand["std"]
            cl = clazz["class"]
            break
                        
                        
    return angle, std, cl, procrustes, coordination

def get_best(results, metal_name):
    coordination = 0
    procrustes = 1

    result = None
    for metal in results:
        if metal["metal"] == metal_name:
            for clazz in metal["ligands"]:
                if clazz["coordination"] < coordination:
                        continue

                if clazz["coordination"] == coordination and  float(clazz["procrustes"]) < procrustes:
                    coordination =  clazz["coordination"]
                    procrustes   =  float(clazz["procrustes"])
                    result = clazz

                if clazz["coordination"] > coordination:
                    coordination =  clazz["coordination"]
                    procrustes   =  float(clazz["procrustes"])
                    result = clazz

    if result is None:
        Logger().info(f"No class for {metal_name} found. Please check the pdb file")
    else:
        Logger().info(f"Best class for {metal_name} is {result['class']} with coordination {coordination} and procrustes {procrustes}")
    return result

   

def get_element_name(mmcif_atom_category, name):
    for i, _ in enumerate(mmcif_atom_category[_atom_id]):
        if _ == name:
            return mmcif_atom_category[_type_symbol][i]

def contains_metal(mmcif_atom_category):

    for element in mmcif_atom_category[_type_symbol]:
        if gemmi.Element(element).is_metal:
            return True
    return False


def update_cif(output_path, path, pdb):
    Logger().info(f"Start processing {path}")
    folder, name = os.path.split(path)
    name = name[:-4]
    folder = os.path.split(folder)[1]
    doc = gemmi.cif.read_file(path)


    if not doc.find_block(f"comp_list"):
        list_block = doc.add_new_block("comp_list", 0)
        x="."
        list_block.set_mmcif_category(_comp_category, {_id: [name], _three_letter_code: [name], _name: [name.lower()], 
            _group: ["."], _number_atoms_all: ["1"], _number_atoms_nh: ["1"], _desc_level: ["."]})

    block = doc.find_block(f"comp_{name}") if doc.find_block(f"comp_{name}") is not None else doc.find_block(f"{name}")
    block.name = f"comp_{name}"

    if block is None:
        Logger().error(f"No block found for {name}|comp_{name}. Please check the cif file.")
        return
    
    atoms = block.get_mmcif_category(_atom_category)
    bonds = block.get_mmcif_category(_bond_category)
    angles = block.get_mmcif_category(_angle_category)
    
    if not atoms:
        Logger().error(f"mmcif category {_atom_category} not found. Please check the cif file.")
        return
    
    new_atoms = dict()
    if _energy not in atoms:
        for key, value in atoms.items():
            new_atoms[key] = value
            if key == _type_symbol:
                new_atoms[_energy] = value
        block.set_mmcif_category(_atom_category, new_atoms)

    

    if not bonds:
        Logger().error(f"mmcif category {_bond_category} not found. Please check the cif file.")
        return
    
    if name in mons and contains_metal(atoms):


        if pdb is None:
            Logger().info(f"Choosing best pdb file")
            pdb = mons[name][0][0]
            Logger().info(f"Best pdb file is {pdb}")

        results = find_classes(name, pdb)


        if len(results) == 0:
            Logger().info(f"No {name} found in {pdb}. Please check the pdb file")
            return
        

    
        best_results = dict()
        for atom_name, element in  zip(atoms[_atom_id], atoms[_type_symbol]):
            if gemmi.Element(element).is_metal:
                best_results[atom_name] = get_best(results, atom_name)

        
        Logger().info(f"Ligand updating started")
        if _value_dist not in bonds:
            bonds[_value_dist] = ["0.00"] * len(bonds[_atom_id_1])
            bonds[_value_dist_esd] = ["0.00"] * len(bonds[_atom_id_1])
            bonds[_value_dist_nucleus] = ["0.00"] * len(bonds[_atom_id_1])
            bonds[_value_dist_nucleus_esd] = ["0.00"] * len(bonds[_atom_id_1])
        
        

        for i, _atoms in enumerate(zip(bonds[_atom_id_1], bonds[_atom_id_2])):
            metal_name, ligand_name = _atoms

            if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal and not gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                continue

            if gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                metal_name, ligand_name = ligand_name, metal_name


            distance, std, _, _, coordination = get_distance(best_results.get(metal_name, None), ligand_name)
            if coordination > 0:
                bonds[_value_dist][i] = bonds[_value_dist_nucleus][i] = str(round(distance, 3))
                bonds[_value_dist_esd][i] = bonds[_value_dist_nucleus_esd][i] = str(round(std, 3))
            

        block.set_mmcif_category(_bond_category, bonds)
        Logger().info(f"Distances updated")


        if not angles:
            Logger().info(f"Creating mmcif category {_angle_category}.")
            angles[_comp_id] = list()
            angles[_atom_id_1] = list()
            angles[_atom_id_2] = list()
            angles[_atom_id_3] = list()
            angles[_value_angle] = list()
            angles[_value_angle_esd] = list()

            for metal_name, clazz in best_results.items():
                if clazz is None:
                    continue
                for ligand in clazz["angles"]:
                    if ligand["ligand1"]["residue"] == name and ligand["ligand2"]["residue"] == name:
                        angles[_comp_id].append(name)
                        angles[_atom_id_1].append(ligand["ligand1"]["name"])
                        angles[_atom_id_2].append(metal_name)
                        angles[_atom_id_3].append(ligand["ligand2"]["name"])
                        angles[_value_angle].append(str(round(ligand["angle"], 3)))
                        angles[_value_angle_esd].append(str(round(ligand["std"], 3)))

        else:
            for i, _atoms in enumerate(zip(angles[_atom_id_1], angles[_atom_id_2], angles[_atom_id_3])):
                ligand1_name, metal_name, ligand2_name = _atoms

                if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal:
                    continue

                angle, std, _, _, coordination = get_angles(best_results.get(metal_name, None), ligand1_name, ligand2_name)
                if coordination > 0:
                    angles[_value_angle][i] = str(round(angle, 3))
                    angles[_value_angle_esd][i] = str(round(std, 3))
            
            
        block.set_mmcif_category(_angle_category, angles)
        Logger().info(f"Angles updated")
        Logger().info(f"Ligand update completed")

        Path(os.path.split(output_path)[0]).mkdir(exist_ok=True, parents=True)
        doc.write_file(output_path)
        report_path = output_path + ".json"
        Logger().info(f"Update written to {output_path}")
        with open(report_path, 'w') as json_file:
            json.dump(results, json_file, 
                                indent=4,  
                                separators=(',',': '))
        Logger().info(f"Report written to {report_path}")
    else:
        if name not in mons :
            Logger().info(f"{name} is not in Ligand - PDB database")
        if not contains_metal(block):
            Logger().info(f"No metal found in {name}")

def get_stats(ligand, pdb, output):
    results = find_classes(ligand, pdb)
    with open(output, 'w') as json_file:
        json.dump(results, json_file, 
                            indent=4,  
                            separators=(',',': '))
    Logger().info(f"Report written to {output}")