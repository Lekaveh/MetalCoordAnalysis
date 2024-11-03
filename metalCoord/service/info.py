
import json
import os
from pathlib import Path
import sys
from metalCoord.analysis.classes import idealClasses
from metalCoord.logging import Logger


d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json"), encoding="utf-8"))


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


def get_coordinations(coordination_num: int = None) -> list:
    """
    Retrieves the ideal coordination classes based on the given coordination number.

    Parameters:
    coordination_num (int): The coordination number to filter the ideal classes. If None, returns all ideal classes.

    Returns:
    list: A list of ideal coordination classes.

    """
    if coordination_num:
        return idealClasses.get_ideal_classes_by_coordination(coordination_num)
    return idealClasses.get_ideal_classes()
