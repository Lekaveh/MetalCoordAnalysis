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


def process_pdbs_list(ligand: str, output: str) -> list:


    """
    Retrieves the PDB files containing the given ligand and optionally writes them to a JSON file.

    output (str): The file path to write the list of PDB files. If not provided, the list will be printed.

    """
    pdbs = get_pdbs_list(ligand)
    if output:
        directory = os.path.dirname(output)
        Path(directory).mkdir(exist_ok=True, parents=True)
        with open(output, 'w', encoding="utf-8") as json_file:
            json.dump(pdbs, json_file,
                    indent=4,
                    separators=(',', ': '))
            Logger().info(f"List of pdbs for {ligand} written to {output}")
    else:
        print(pdbs)

def get_coordinations(coordination_num: int = None, metal: str = None, cod: bool = False) -> list:
    """
    Retrieve coordination information based on the provided parameters.
    Args:
        coordination_num (int, optional): The coordination number to filter by. Defaults to None.
        metal (str, optional): The metal element to filter by. Defaults to None.
        cod (bool, optional): Whether to include the COD IDs in the result. Defaults to False.
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
        return DB.get_frequency_coordination(coordination_num, cod=cod)
        
    if metal and coordination_num:
        return DB.get_frequency_metal_ccordination(metal, coordination_num, cod=cod)
    
    if metal and not coordination_num:
        return DB.get_frequency_metal(metal, cod=cod)
    
    return DB.get_frequency(cod=cod)

def process_coordinations(coordination_num: int = None, metal: str = None, output: str = None, cod: bool = False) -> None:
    """
    Retrieve coordination information based on the provided parameters and optionally write it to a JSON file.
    Args:
        coordination_num (int, optional): The coordination number to filter by. Defaults to None.
        metal (str, optional): The metal element to filter by. Defaults to None.
        output (str, optional): The file path to write the coordination information. Defaults to None.
        cod (bool, optional): Whether to include the COD IDs in the result. Defaults to False.
    Raises:
        ValueError: If the provided metal is not a valid metal element.
    Notes:
        - If only `coordination_num` is provided, writes ideal classes by coordination number to a JSON file.
        - If both `metal` and `coordination_num` are provided, writes frequency data for the metal and coordination number to a JSON file.
        - If only `metal` is provided, writes frequency data for the metal across all coordination numbers to a JSON file.
        - If neither `coordination_num` nor `metal` is provided, writes all ideal classes to a JSON file.
    """
    coordinations = get_coordinations(coordination_num, metal, cod = cod)
    if output:
        directory = os.path.dirname(output)
        Path(directory).mkdir(exist_ok=True, parents=True)
        with open(output, 'w', encoding="utf-8") as json_file:
            json.dump(coordinations, json_file,
                    indent=4,
                    separators=(',', ': '))
            Logger().info(f"Coordinations info written to {output}")
    else:
        print(coordinations)
