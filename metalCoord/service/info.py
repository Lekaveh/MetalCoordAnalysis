import json
import os
from pathlib import Path
import sys
import gemmi
from metalCoord.analysis.classes import idealClasses
from metalCoord.analysis.data import DB
from metalCoord.logging import Logger



d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json"), encoding="utf-8"))


def get_metal_ligand_list() -> list:
    """
    Retrieves the list of metals and their corresponding ligands.

    Returns:
    list: A list of metals and their corresponding ligands.

    """
    return mons.keys()

def get_pdbs_list(ligand: str) -> list:
    """
    Retrieves the PDB files containing the given ligand.

    Parameters:
    ligand (str): The name of the ligand.

    Returns:
    list: A list of PDB files containing the ligand.

    """
    if ligand in mons:
        return {ligand: sorted(mons[ligand], key=lambda x: (not x[2], x[1] if x[1] else 10000))}
    return {ligand: []}


def save_pdbs_list(ligand: str, output: str) -> list:
    """
    Retrieves the PDB files containing the given ligand.

    Parameters:
    ligand (str): The name of the ligand.

    Returns:
    list: A list of PDB files containing the ligand.

    """
    pdbs = get_pdbs_list(ligand)
    directory = os.path.dirname(output)
    Path(directory).mkdir(exist_ok=True, parents=True)
    with open(output, 'w', encoding="utf-8") as json_file:
        json.dump(pdbs, json_file,
                  indent=4,
                  separators=(',', ': '))
        Logger().info(f"List of pdbs for {ligand} written to {output}")


def get_coordinations(coordination_num: int = None, metal: str = None) -> list:
    """
    Retrieve coordination information based on the provided parameters.
    Args:
        coordination_num (int, optional): The coordination number to filter by. Defaults to None.
        metal (str, optional): The metal element to filter by. Defaults to None.
    Returns:
        list: A list of coordination information based on the provided parameters.
    Raises:
        ValueError: If the provided metal is not a valid metal element.
    Notes:
        - If only `coordination_num` is provided, returns ideal classes by coordination number.
        - If both `metal` and `coordination_num` are provided, returns frequency data for the metal and coordination number.
        - If only `metal` is provided, returns frequency data for the metal across all coordination numbers.
        - If neither `coordination_num` nor `metal` is provided, returns all ideal classes.
    """

    if metal:
        if not gemmi.Element(metal).is_metal:
            raise ValueError(f"{metal} is not a metal element.")
        metal = metal.lower().capitalize()
    if coordination_num and not metal:
        return idealClasses.get_ideal_classes_by_coordination(coordination_num)
        
    if metal and coordination_num:
        return DB.get_frequency(metal, coordination_num)
    
    if metal and not coordination_num:
        return DB.get_frequency_all(metal)
    
    return idealClasses.get_ideal_classes()
