import re
import json
import os
from pathlib import Path
import sys
import gemmi
from metalCoord.analysis.stats import find_classes
from metalCoord.logging import Logger
import networkx as nx



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

def code(ligand1_name, metal_name, ligand2_name):
    return ''.join(sorted([ligand1_name, metal_name, ligand2_name]))  

def get_element_name(mmcif_atom_category, name):
    for i, _ in enumerate(mmcif_atom_category[_atom_id]):
        if _ == name:
            return mmcif_atom_category[_type_symbol][i]

def contains_metal(mmcif_atom_category):

    for element in mmcif_atom_category[_type_symbol]:
        if gemmi.Element(element).is_metal:
            return True
    return False



def find_minimal_cycles(bonds):
    graph = nx.Graph()
    graph.add_edges_from ([[a1, a2] for a1, a2 in bonds])    
    # Find the cycle basis of the graph
    cycle_basis = nx.simple_cycles (graph)

    return cycle_basis

def bond_exist(bonds, atom1, atom2):
    for b1, b2 in zip(bonds[_atom_id_1], bonds[_atom_id_2]):
        if (b1 == atom1 and b2 == atom2) or (b1 == atom2 and b2 == atom1):
            return True
    return False

def update_cif(output_path, path, pdb):
    Logger().info(f"Start processing {path}")
    folder, name = os.path.split(path)
    folder = os.path.split(folder)[1]
    doc = gemmi.cif.read_file(path)


    name = None    
    for block in doc:
        matches = re.findall(r"^(?:comp_)?([A-Za-z0-9]{3}$)", block.name)
        if matches:
            name = matches[0]
            Logger().info(f"Ligand {name} found")
            break

    if not name:
        Logger().error(f"No block found for <name>|comp_<name>. Please check the cif file.")
        return

    if not doc.find_block(f"comp_list"):
        list_block = doc.add_new_block("comp_list", 0)
        x="."
        list_block.set_mmcif_category(_comp_category, {_id: [name], _three_letter_code: [name], _name: [name.lower()], 
            _group: ["."], _number_atoms_all: ["1"], _number_atoms_nh: ["1"], _desc_level: ["."]})


    block = doc.find_block(f"comp_{name}") if doc.find_block(f"comp_{name}") is not None else doc.find_block(f"{name}")
    

    if block is None:
        Logger().error(f"No block found for {name}|comp_{name}. Please check the cif file.")
        return
    
    block.name = f"comp_{name}"
    
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
    
    
    if  contains_metal(atoms):

        if pdb is None:
            if name not in mons:
                Logger().info(f"There is no pdb in our Ligand-PDB database. Please specify the pdb file")
                return
            Logger().info(f"Choosing best pdb file")
            all_candidates = sorted(mons[name], key=lambda x: x[1])

            candidates = [mon for mon in all_candidates if mon[2]]
            
            if len(candidates) == 0:
                mon = all_candidates[0]
            else:
                mon = candidates[0]
            if mon[1] > 2:
                if len(candidates) == 0:
                    Logger().warning(f"There is no pdb with necesarry resolution and occupancy in our Ligand-PDB database. Please specify the pdb file")       
                else:
                    Logger().warning(f"There is no pdb with necesarry resolution in our Ligand-PDB database. Please specify the pdb file")   
            else:
                if len(candidates) == 0:
                    Logger().warning(f"There is no pdb with necesarry occupancy in our Ligand-PDB database. Please specify the pdb file")   

            pdb = mon[0]
            Logger().info(f"Best pdb file is {pdb}")

        def get_bonds(atoms, bonds):
            result = {}
            for atom1, atom2  in zip(bonds[_atom_id_1], bonds[_atom_id_2]):
                if not gemmi.Element(get_element_name(atoms, atom1)).is_metal and not gemmi.Element(get_element_name(atoms, atom2)).is_metal:
                    continue
                if  gemmi.Element(get_element_name(atoms, atom1)).is_metal and gemmi.Element(get_element_name(atoms, atom2)).is_metal:
                    continue

                if gemmi.Element(get_element_name(atoms, atom2)).is_metal:
                    atom1, atom2 = atom2, atom1

                result.setdefault(atom1, []).append(atom2)
            
            return result
                    
        
        pdbStats = find_classes(name, pdb, get_bonds(atoms, bonds))


        if pdbStats.isEmpty():
            # Logger().info(f"No coordination found for {name}  in {pdb}. Please check the pdb file")
            return
        
        
        Logger().info(f"Ligand updating started")
        if _value_dist not in bonds:
            bonds[_value_dist] = [None for x in range(len(bonds[_atom_id_1]))]
            bonds[_value_dist_esd] = [None for x in range(len(bonds[_atom_id_1]))]
            bonds[_value_dist_nucleus] = [None for x in range(len(bonds[_atom_id_1]))]
            bonds[_value_dist_nucleus_esd] = [None for x in range(len(bonds[_atom_id_1]))]
        
        

        for i, _atoms in enumerate(zip(bonds[_atom_id_1], bonds[_atom_id_2])):
            metal_name, ligand_name = _atoms

            if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal and not gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                continue

            if gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                metal_name, ligand_name = ligand_name, metal_name


            bondStat = pdbStats.getLigandDistance(metal_name, ligand_name)
            if  bondStat:
                bonds[_value_dist][i] = bonds[_value_dist_nucleus][i] = str(round(bondStat.distance[0], 3))
                bonds[_value_dist_esd][i] = bonds[_value_dist_nucleus_esd][i] = str(round(bondStat.std[0], 3))
            

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

            for metal_name in pdbStats.metalNames():
                for angle in pdbStats.getLigandAngles(metal_name):
                    if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                        continue
                    angles[_comp_id].append(name)
                    angles[_atom_id_1].append(angle.ligand1.name)
                    angles[_atom_id_2].append(metal_name)
                    angles[_atom_id_3].append(angle.ligand2.name)
                    angles[_value_angle].append(str(round(angle.angle, 3)))
                    angles[_value_angle_esd].append(str(round(angle.std, 3)))

        else:
            update_angles = []
            for i, _atoms in enumerate(zip(angles[_atom_id_1], angles[_atom_id_2], angles[_atom_id_3])):
                ligand1_name, metal_name, ligand2_name = _atoms

                if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal:
                    continue

                angleStat = pdbStats.getLigandAngle(metal_name, ligand1_name, ligand2_name)
                if angleStat:
                    angles[_value_angle][i] = str(round(angleStat.angle, 3))
                    angles[_value_angle_esd][i] = str(round(angleStat.std, 3))
                    update_angles.append(code(ligand1_name, metal_name, ligand2_name))
            
            
            for metal_name in pdbStats.metalNames():
                for angle in pdbStats.getLigandAngles(metal_name):
                    if code(angle.ligand1.name, metal_name, angle.ligand2.name) in update_angles:
                        continue

                    if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                        continue
                    angles[_comp_id].append(name)
                    angles[_atom_id_1].append(angle.ligand1.name)
                    angles[_atom_id_2].append(metal_name)
                    angles[_atom_id_3].append(angle.ligand2.name)
                    angles[_value_angle].append(str(round(angle.angle, 3)))
                    angles[_value_angle_esd].append(str(round(angle.std, 3)))
            
        v = []
        monomer = list(pdbStats.monomers())[-1]
        for metalStat in monomer.metals:
            for bond in metalStat.getAllDistances():
                v.append((metalStat.code, bond.ligand.code))
        
        for cycle in find_minimal_cycles(v):
            if len(cycle) == 4:
                if gemmi.Element(cycle[0][1]).is_metal:
                    metal1 = cycle[0]
                    metal2 = cycle[2]
                    ligand1 = cycle[1]
                    ligand2 = cycle[3]
                else:
                    metal1 = cycle[1]
                    metal2 = cycle[3]
                    ligand1 = cycle[0]
                    ligand2 = cycle[2]
                
                              
                angle1 = monomer.getAngle(metal1[0], ligand1, ligand2)
                angle2 = monomer.getAngle(metal2[0], ligand1, ligand2)

                if not angle1 or not angle2:
                    if not angle1:
                        Logger().warning(f"Angle {ligand1[0]}- {metal1[0]}-{ligand2[0]} not found in {monomer.code}")

                    if not angle2:
                        Logger().warning(f"Angle {ligand1[0]}- {metal2[0]}-{ligand2[0]} not found in {monomer.code}")
                    continue

                val = (360 - angle1.angle - angle2.angle)/2
                std = 5.0

                for ligand in [ligand1, ligand2]:
                    if monomer.code == ligand[2:]:
                        found = False
                        for i, _atoms in enumerate(zip(angles[_atom_id_1], angles[_atom_id_2], angles[_atom_id_3])):
                            metal1_name, ligand_name, metal2_name = _atoms
                            if (metal1_name == metal1[0] and ligand_name == ligand[0] and metal2_name == metal2[0]) or (metal1_name == metal2[0] and ligand_name == ligand[0] and metal2_name == metal1[0]) :
                                angles[_value_angle][i] = str(round(val, 3))
                                angles[_value_angle_esd][i] = str(round(std, 3))
                                found = True
                                break
                        if not found:
                            angles[_comp_id].append(name)
                            angles[_atom_id_1].append(metal1[0])
                            angles[_atom_id_2].append(ligand[0])
                            angles[_atom_id_3].append(metal2[0])
                            angles[_value_angle].append(str(round(val, 3)))
                            angles[_value_angle_esd].append(str(round(std, 3)))       

        block.set_mmcif_category(_angle_category, angles)
        Logger().info(f"Angles updated")
        Logger().info(f"Ligand update completed")

        Path(os.path.split(output_path)[0]).mkdir(exist_ok=True, parents=True)
        doc.write_file(output_path)
        report_path = output_path + ".json"
        Logger().info(f"Update written to {output_path}")
        with open(report_path, 'w') as json_file:
            json.dump(pdbStats.json(), json_file, 
                                indent=4,  
                                separators=(',',': '))
        Logger().info(f"Report written to {report_path}")
    else:
        Logger().info(f"No metal found in {name}")

def get_stats(ligand, pdb, output):
    results = find_classes(ligand, pdb).json()
    with open(output, 'w') as json_file:
        json.dump(results, json_file, 
                            indent=4,  
                            separators=(',',': '))
    Logger().info(f"Report written to {output}")