import re
import json
import os
from pathlib import Path
import sys
import gemmi
import networkx as nx
import metalCoord
from metalCoord.analysis.classify import find_classes_pdb, find_classes_cif
from metalCoord.analysis.models import PdbStats
from metalCoord.config import Config
from metalCoord.logging import Logger
from metalCoord.cif.utils import (
    ACEDRG_CATEGORY, ATOM_CATEGORY, BOND_CATEGORY, ANGLE_CATEGORY, COMP_CATEGORY, COMP_ID,
    PROGRAM_NAME, PROGRAM_VERSION, TYPE, ATOM_ID, TYPE_SYMBOL, VALUE_DIST, VALUE_DIST_ESD,
    VALUE_DIST_NUCLEUS, VALUE_DIST_NUCLEUS_ESD, VALUE_ANGLE, VALUE_ANGLE_ESD, ATOM_ID_1, ATOM_ID_2, ATOM_ID_3,
    ID, THREE_LETTER_CODE, NAME, GROUP, NUMBER_ATOMS_ALL, NUMBER_ATOMS_NH, DESC_LEVEL, ENERGY
)
from metalCoord.cif.utils import get_element_name, get_bonds

d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json"), encoding="utf-8"))


def decompose(values, n):
    """
    Decompose a list of values into sublists of size n.

    Args:
        values (list): The list of values to be decomposed.
        n (int): The size of each sublist.

    Returns:
        list: A list of sublists, each containing n values.
    """
    result = []
    for i in range(0, len(values)//n):
        result.append(values[i*n:i*n + n])
    return result


def pack(values):
    """
    Pack a list of values into sublists, grouping them by index.

    Args:
        values (list): The list of values to be packed.

    Returns:
        list: A list of sublists, each containing the values at the same index.
    """
    return [list(x) for x in zip(*values)]


def get_distance(clazz, ligand_name):
    """
    Get the distance, standard deviation, class, procrustes value, and coordination
    for a given ligand name in a coordination class.

    Parameters:
    - clazz (dict): The coordination class containing ligand information.
    - ligand_name (str): The name of the ligand to retrieve information for.

    Returns:
    - distance (float): The distance of the ligand.
    - std (float): The standard deviation of the ligand distance.
    - cl (str): The class of the coordination.
    - procrustes (float): The procrustes value of the coordination.
    - coordination (int): The coordination number of the coordination.

    If clazz is None, all return values will be set to their default values.
    """
    coordination = 0
    cl = ""
    procrustes = 1
    distance = -1
    std = -1

    if clazz is None:
        return distance, std, cl, procrustes, coordination

    for ligand in clazz["base"]:
        if ligand["ligand"]["name"] == ligand_name:
            coordination = clazz["coordination"]
            procrustes = float(clazz["procrustes"])
            distance = ligand["distance"]
            std = ligand["std"]
            cl = clazz["class"]
            break

    return distance, std, cl, procrustes, coordination


def get_angles(clazz, ligand_name1, ligand_name2):
    """
    Get the angle, standard deviation, class, procrustes value, and coordination
    for a given pair of ligand names in a coordination class.

    Parameters:
    - clazz (dict): The coordination class containing ligand information.
    - ligand_name1 (str): The name of the first ligand.
    - ligand_name2 (str): The name of the second ligand.

    Returns:
    - angle (float): The angle between the ligands.
    - std (float): The standard deviation of the angle.
    - cl (str): The class of the coordination.
    - procrustes (float): The procrustes value of the coordination.
    - coordination (int): The coordination number of the coordination.

    If clazz is None, all return values will be set to their default values.
    """
    coordination = 0
    cl = ""
    procrustes = 1
    angle = -1
    std = -1

    if clazz is None:
        return angle, std, cl, procrustes, coordination

    for ligand in clazz["angles"]:
        if ((ligand["ligand1"]["name"] == ligand_name1) and (ligand["ligand2"]["name"] == ligand_name2)) or ((ligand["ligand2"]["name"] == ligand_name1) and (ligand["ligand1"]["name"] == ligand_name2)):
            coordination = clazz["coordination"]
            procrustes = float(clazz["procrustes"])
            angle = ligand["angle"]
            std = ligand["std"]
            cl = clazz["class"]
            break

    return angle, std, cl, procrustes, coordination


def code(ligand1_name, metal_name, ligand2_name):
    """
    Generate a code by sorting and concatenating the ligand names.

    Parameters:
    - ligand1_name (str): The name of the first ligand.
    - metal_name (str): The name of the metal.
    - ligand2_name (str): The name of the second ligand.

    Returns:
    - code (str): The generated code.
    """
    return ''.join(sorted([ligand1_name, metal_name, ligand2_name]))




def contains_metal(mmcif_atom_category):
    """
    Check if the mmcif atom category contains any metal atoms.

    Parameters:
    - mmcif_atom_category (dict): The mmcif atom category.

    Returns:
    - contains_metal (bool): True if the category contains metal atoms, False otherwise.
    """
    for element in mmcif_atom_category[TYPE_SYMBOL]:
        if gemmi.Element(element).is_metal:
            return True
    return False


def find_minimal_cycles(bonds):
    """
    Find the minimal cycles in a graph represented by a list of bonds.

    Parameters:
    - bonds (dict): The dictionary containing the atom IDs of the bonds.

    Returns:
    - cycle_basis (list): A list of minimal cycles in the graph.
    """
    graph = nx.Graph()
    graph.add_edges_from([[a1, a2] for a1, a2 in bonds])
    # Find the cycle basis of the graph
    cycle_basis = nx.simple_cycles(graph, length_bound=4)

    return cycle_basis


def bond_exist(bonds, atom1, atom2):
    """
    Check if a bond exists between two atoms.

    Parameters:
    - bonds (dict): The dictionary containing the atom IDs of the bonds.
    - atom1 (str): The ID of the first atom.
    - atom2 (str): The ID of the second atom.

    Returns:
    - bond_exists (bool): True if the bond exists, False otherwise.
    """
    for b1, b2 in zip(bonds[ATOM_ID_1], bonds[ATOM_ID_2]):
        if (b1 == atom1 and b2 == atom2) or (b1 == atom2 and b2 == atom1):
            return True
    return False


def save_cods(pdb_stats: PdbStats, path: str):
    """
    Save COD files for ligands in the given PdbStats object.

    Args:
        pdb_stats (PdbStats): The PdbStats object containing the ligand information.
        path (str): The path where the COD files will be saved.

    Returns:
        None
    """
    output = os.path.join(path, "cod")
    for monomer in pdb_stats.monomers():
        for metal_stats in monomer.metals:
            for ligand_stats in metal_stats.ligands:
                output_folder = os.path.join(output, metal_stats.residue, metal_stats.chain, str(
                    metal_stats.sequence), metal_stats.metal, ligand_stats.clazz)
                Path(output_folder).mkdir(exist_ok=True, parents=True)
                for file, st in ligand_stats.cods:
                    st.write_pdb(os.path.join(output_folder, file))




def update_cif(output_path, path, pdb, use_cif=False, clazz = None):
    """
    Update the CIF file with ligand information.

    Args:
    - output_path (str): The path where the updated CIF file will be saved.
    - path (str): The path to the original CIF file.
    - pdb (str): The path to the PDB file.
    - use_cif (bool): Whether to use the CIF file for classification.
    - clazz (str): Predefined class.

    Returns:
    - None
    """
    Logger().info(f"Start processing {path}")
    folder, name = os.path.split(path)
    folder = os.path.split(folder)[1]
    doc = gemmi.cif.read_file(path)

    name = None
    for block in doc:
        matches = re.findall(r"^(?:comp_)?([A-Za-z0-9]{3,}$)", block.name)
        if matches:
            name = matches[0]
            if name == 'list':
                continue
            Logger().info(f"Ligand {name} found")
            break

    if not name:
        raise ValueError(
            "No block found for <name>|comp_<name>. Please check the CIF file.")

    block = doc.find_block(f"comp_{name}") if doc.find_block(
        f"comp_{name}") is not None else doc.find_block(f"{name}")

    if block is None:
        raise ValueError(
            f"No block found for {name}|comp_{name}. Please check the CIF file.")

    block.name = f"comp_{name}"

    ace_drg = block.get_mmcif_category(ACEDRG_CATEGORY)
    if not ace_drg:
        ace_drg[COMP_ID] = list()
        ace_drg[PROGRAM_NAME] = list()
        ace_drg[PROGRAM_VERSION] = list()
        ace_drg[TYPE] = list()

    ace_drg[COMP_ID].append(name)
    ace_drg[PROGRAM_NAME].append("metalCoord")
    ace_drg[PROGRAM_VERSION].append(metalCoord.__version__)
    ace_drg[TYPE].append("metal coordination analysis")

    block.set_mmcif_category(ACEDRG_CATEGORY, ace_drg)

    atoms = block.get_mmcif_category(ATOM_CATEGORY)
    bonds = block.get_mmcif_category(BOND_CATEGORY)
    angles = block.get_mmcif_category(ANGLE_CATEGORY)

    if not atoms:
        raise ValueError(
            f"mmcif category {ATOM_CATEGORY} not found. Please check the CIF file.")

    n_atoms = len(atoms[ATOM_ID])
    n_nhatoms = len([x for x in atoms[TYPE_SYMBOL] if x != "H"])

    new_atoms = dict()
    if ENERGY not in atoms:
        for key, value in atoms.items():
            new_atoms[key] = value
            if key == TYPE_SYMBOL:
                new_atoms[ENERGY] = value
        block.set_mmcif_category(ATOM_CATEGORY, new_atoms)

    if not bonds:
        # raise Exception(f"mmcif category {BOND_CATEGORY} not found. Please check the CIF file.")
        Logger().warning(
            f"mmcif category {BOND_CATEGORY} not found. Please check the CIF file.")
    if contains_metal(atoms):
        if use_cif:
            pdb_stats = find_classes_cif(name, atoms, bonds, clazz=clazz)
        else:
            if pdb is None:
                if name not in mons:
                    raise Exception(
                        "There is no PDB in our Ligand-PDB database. Please specify the PDB file")
                Logger().info("Choosing best PDB file")
                all_candidates = sorted(mons[name], key=lambda x: (
                    not x[2], x[1] if x[1] else 10000))

                candidates = [mon for mon in all_candidates if mon[2]]

                if len(candidates) == 0:
                    mon = all_candidates[0]
                else:
                    mon = candidates[0]
                if mon[1] and mon[1] > 2:
                    if len(candidates) == 0:
                        Logger().warning("There is no PDB with necessary resolution and occupancy in our Ligand-PDB database. Please specify the PDB file")
                    else:
                        Logger().warning("There is no PDB with necessary resolution in our Ligand-PDB database. Please specify the PDB file")
                else:
                    if len(candidates) == 0:
                        Logger().warning("There is no PDB with necessary occupancy in our Ligand-PDB database. Please specify the PDB file")

                pdb = mon[0]
                Logger().info(f"Best PDB file is {pdb}")

            pdb_stats = find_classes_pdb(
                name, pdb, get_bonds(atoms, bonds), only_best=True, clazz = clazz)

        if pdb_stats.is_empty():
            # Logger().info(f"No coordination found for {name}  in {pdb}. Please check the PDB file")
            raise ValueError(
                f"No coordination found for {name}  in {pdb}. Please check the PDB file")

        Logger().info("Ligand updating started")
        if bonds:
            if VALUE_DIST not in bonds:
                bonds[VALUE_DIST] = [
                    None for x in range(len(bonds[ATOM_ID_1]))]
                bonds[VALUE_DIST_ESD] = [
                    None for x in range(len(bonds[ATOM_ID_1]))]
                bonds[VALUE_DIST_NUCLEUS] = [
                    None for x in range(len(bonds[ATOM_ID_1]))]
                bonds[VALUE_DIST_NUCLEUS_ESD] = [
                    None for x in range(len(bonds[ATOM_ID_1]))]

            for i, _atoms in enumerate(zip(bonds[ATOM_ID_1], bonds[ATOM_ID_2])):
                metal_name, ligand_name = _atoms

                if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal and not gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                    continue

                if gemmi.Element(get_element_name(atoms, ligand_name)).is_metal:
                    metal_name, ligand_name = ligand_name, metal_name

                bondStat = pdb_stats.get_ligand_distance(
                    metal_name, ligand_name)
                if bondStat:
                    bonds[VALUE_DIST][i] = bonds[VALUE_DIST_NUCLEUS][i] = str(
                        round(bondStat.distance[0], 3))
                    bonds[VALUE_DIST_ESD][i] = bonds[VALUE_DIST_NUCLEUS_ESD][i] = str(
                        round(bondStat.std[0], 3))

            block.set_mmcif_category(BOND_CATEGORY, bonds)
        Logger().info("Distances updated")

        if not angles:
            Logger().info("Creating mmcif category {ANGLE_CATEGORY}.")
            angles[COMP_ID] = list()
            angles[ATOM_ID_1] = list()
            angles[ATOM_ID_2] = list()
            angles[ATOM_ID_3] = list()
            angles[VALUE_ANGLE] = list()
            angles[VALUE_ANGLE_ESD] = list()

            for metal_name in pdb_stats.metal_names():
                for angle in pdb_stats.get_ligand_angles(metal_name):
                    if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                        continue
                    angles[COMP_ID].append(name)
                    angles[ATOM_ID_1].append(angle.ligand1.name)
                    angles[ATOM_ID_2].append(metal_name)
                    angles[ATOM_ID_3].append(angle.ligand2.name)
                    angles[VALUE_ANGLE].append(str(round(angle.angle, 3)))
                    angles[VALUE_ANGLE_ESD].append(str(round(angle.std, 3)))

        else:
            update_angles = []
            for i, _atoms in enumerate(zip(angles[ATOM_ID_1], angles[ATOM_ID_2], angles[ATOM_ID_3])):
                ligand1_name, metal_name, ligand2_name = _atoms

                if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal:
                    continue

                angle_stat = pdb_stats.get_ligand_angle(
                    metal_name, ligand1_name, ligand2_name)
                if angle_stat:
                    angles[VALUE_ANGLE][i] = str(round(angle_stat.angle, 3))
                    angles[VALUE_ANGLE_ESD][i] = str(round(angle_stat.std, 3))
                    update_angles.append(
                        code(ligand1_name, metal_name, ligand2_name))

            for metal_name in pdb_stats.metal_names():
                for angle in pdb_stats.get_ligand_angles(metal_name):
                    if code(angle.ligand1.name, metal_name, angle.ligand2.name) in update_angles:
                        continue

                    if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                        continue
                    angles[COMP_ID].append(name)
                    angles[ATOM_ID_1].append(angle.ligand1.name)
                    angles[ATOM_ID_2].append(metal_name)
                    angles[ATOM_ID_3].append(angle.ligand2.name)
                    angles[VALUE_ANGLE].append(str(round(angle.angle, 3)))
                    angles[VALUE_ANGLE_ESD].append(str(round(angle.std, 3)))

        v = []
        monomer = list(pdb_stats.monomers())[-1]
        for metal_stat in monomer.metals:
            for bond in metal_stat.get_all_distances():
                v.append((metal_stat.code, bond.ligand.code))

        Logger().info("update cycles")
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

                angle1 = monomer.get_angle(metal1[0], ligand1, ligand2)
                angle2 = monomer.get_angle(metal2[0], ligand1, ligand2)

                if not angle1 or not angle2:
                    if not angle1:
                        Logger().warning(
                            f"Angle {ligand1[0]}- {metal1[0]}-{ligand2[0]} not found in {monomer.code}")

                    if not angle2:
                        Logger().warning(
                            f"Angle {ligand1[0]}- {metal2[0]}-{ligand2[0]} not found in {monomer.code}")
                    continue

                val = (360 - angle1.angle - angle2.angle)/2
                std = 5.0

                for ligand in [ligand1, ligand2]:
                    if monomer.code == ligand[2:]:
                        found = False
                        for i, _atoms in enumerate(zip(angles[ATOM_ID_1], angles[ATOM_ID_2], angles[ATOM_ID_3])):
                            metal1_name, ligand_name, metal2_name = _atoms
                            if (metal1_name == metal1[0] and ligand_name == ligand[0] and metal2_name == metal2[0]) or (metal1_name == metal2[0] and ligand_name == ligand[0] and metal2_name == metal1[0]):
                                angles[VALUE_ANGLE][i] = str(round(val, 3))
                                angles[VALUE_ANGLE_ESD][i] = str(round(std, 3))
                                found = True
                                break
                        if not found:
                            angles[COMP_ID].append(name)
                            angles[ATOM_ID_1].append(metal1[0])
                            angles[ATOM_ID_2].append(ligand[0])
                            angles[ATOM_ID_3].append(metal2[0])
                            angles[VALUE_ANGLE].append(str(round(val, 3)))
                            angles[VALUE_ANGLE_ESD].append(str(round(std, 3)))

        block.set_mmcif_category(ANGLE_CATEGORY, angles)
        Logger().info("Angles updated")

        if not doc.find_block("comp_list"):
            list_block = doc.add_new_block("comp_list", 0)
            x = "."
            list_block.set_mmcif_category(COMP_CATEGORY, {ID: [name], THREE_LETTER_CODE: [name], NAME: [name.lower()],
                                                          GROUP: ["."], NUMBER_ATOMS_ALL: [str(n_atoms)], NUMBER_ATOMS_NH: [str(n_nhatoms)], DESC_LEVEL: ["."]})

        Logger().info("Ligand update completed")

        Path(os.path.split(output_path)[0]).mkdir(exist_ok=True, parents=True)
        doc.write_file(output_path)
        report_path = output_path + ".json"
        Logger().info(f"Update written to {output_path}")
        directory = os.path.dirname(output_path)
        Path(directory).mkdir(exist_ok=True, parents=True)
        with open(report_path, 'w', encoding="utf-8") as json_file:
            json.dump(pdb_stats.json(), json_file,
                      indent=4,
                      separators=(',', ': '))

        if Config().save:
            save_cods(pdb_stats, os.path.dirname(output_path))
        Logger().info(f"Report written to {report_path}")
    else:
        Logger().info(f"No metal found in {name}")


def get_stats(ligand, pdb, output, clazz = None):
    """
    Retrieves statistics for a given ligand and PDB file and writes the results to a JSON file.

    Args:
        ligand (str): The name of the ligand.
        pdb (str): The path to the PDB file.
        output (str): The path to the output JSON file.
        clazz (str): Predefined class.

    Returns:
        None
    """
    pdb_stats = find_classes_pdb(ligand, pdb, clazz=clazz)
    results = pdb_stats.json()

    directory = os.path.dirname(output)
    Path(directory).mkdir(exist_ok=True, parents=True)
    with open(output, 'w', encoding="utf-8") as json_file:
        json.dump(results, json_file,
                  indent=4,
                  separators=(',', ': '))
    if Config().save:
        save_cods(pdb_stats, os.path.dirname(output))

    Logger().info(f"Report written to {output}")
