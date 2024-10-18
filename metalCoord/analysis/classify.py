
import os
import gemmi
from tqdm import tqdm
from metalCoord.analysis.classes import Classificator
from metalCoord.analysis.data import DB
from metalCoord.analysis.cod import StrictCandidateFinder, ElementCandidateFinder, ElementInCandidateFinder, AnyElementCandidateFinder, NoCoordinationCandidateFinder, CovalentCandidateFinder
from metalCoord.analysis.stats import OnlyDistanceStatsFinder, WeekCorrespondenceStatsFinder, StrictCorrespondenceStatsFinder, CovalentStatsFinder
from metalCoord.analysis.models import MetalStats, PdbStats
from metalCoord.analysis.structures import get_ligands, Ligand
from metalCoord.load.rcsb import load_pdb
from metalCoord.logging import Logger



def get_structures(ligand, path, bonds=None, only_best=False) -> list[Ligand]:
    """
    Retrieves structures contained in a specific ligand from a given file or 4-letter PDB code.

    Args:
        ligand (str): The name of the ligand to search for.
        path (str): The path to the file or 4-letter PDB code.
        bonds (dict, optional): A dictionary of bond information. Defaults to None.
        only_best (bool, optional): Flag indicating whether to return only the best structure for each metal. Defaults to False.

    Returns:
        list: A list of structures contained in the specified ligand.

    Raises:
        Exception: If an existing pdb or mmcif file path should be provided or a 4-letter pdb code.
    """
    if bonds is None:
        bonds = {}
    # Rest of the function code...

    if os.path.isfile(path):
        st = gemmi.read_structure(path)
        return get_ligands(st, ligand, bonds, only_best=only_best)

    elif len(path) == 4:
        pdb, file_type = load_pdb(path)
        if file_type == 'cif':
            cif_block = gemmi.cif.read_string(pdb)[0]
            st = gemmi.make_structure_from_block(cif_block)
        else:
            st = gemmi.read_pdb_string(pdb)
        return get_ligands(st, ligand, bonds, only_best=only_best)

    else:
        raise FileNotFoundError(
            "Existing pdb or mmcif file path should be provided or 4 letter pdb code")


convalent_strategy = CovalentStatsFinder(CovalentCandidateFinder())
strategies = [StrictCorrespondenceStatsFinder(StrictCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementInCandidateFinder()),
              WeekCorrespondenceStatsFinder(AnyElementCandidateFinder()),
              OnlyDistanceStatsFinder(NoCoordinationCandidateFinder()),
              convalent_strategy
              ]


def find_classes(ligand: str, pdb_name: str, bonds: dict = None, only_best: bool = False) -> PdbStats:
    """
    Analyzes structures in a given PDB file for patterns and returns the statistics for ligands (metals) found.

    Args:
        ligand (str): The name of the ligand.
        pdb_name (str): The name of the PDB file.
        bonds (dict, optional): A dictionary of bonds. Defaults to None.
        only_best (bool, optional): Flag indicating whether to consider only the best structures. Defaults to False.

    Returns:
        PdbStats: An object containing the statistics for ligands (metals) found.
    """
    if bonds is None:
        bonds = {}
    Logger().info(f"Analyzing structures in {pdb_name} for patterns")
    structures = get_structures(ligand, pdb_name, bonds, only_best)
    for structure in tqdm(structures, disable=~Logger().progress_bars):
        Logger().info(
            f"Structure for {structure} found. Coordination number: {structure.coordination()}. {structure.name_code_with_symmetries()}")
    Logger().info(f"{len(structures)} structures found.")
    results = PdbStats()
    classificator = Classificator()

    classes = {}
    for structure in tqdm(structures, desc="Structures", position=0, disable=not Logger().progress_bars):
        structure_classes = []

        for class_result in tqdm(classificator.classify(structure), desc="Coordination", position=1, leave=False, disable=not Logger().progress_bars):
            structure_classes.append(class_result)

        if not structure_classes:
            new_structure = structure.clean_the_farthest(len(bonds)) 
            for class_result in tqdm(classificator.classify(new_structure), desc="Coordination", position=1, leave=False, disable=not Logger().progress_bars):
                structure_classes.append(class_result)
            if structure_classes:
                structure = new_structure
        classes[structure] = structure_classes

    for structure, structure_classes in classes.items():
        candidantes = []
        for class_result in structure_classes:
            candidantes.append(class_result.clazz)
        Logger().info(f"Candidates for {structure} : {candidantes}")

    for structure in tqdm(classes.keys(), desc="Structures", position=0, disable=not Logger().progress_bars):
        metal_stats = MetalStats(structure)
        if classes[structure]:
            for class_result in classes[structure]:
                for strategy in tqdm(strategies, desc="Strategies", position=1, leave=False, disable=not Logger().progress_bars):
                    ligand_stats = strategy.get_stats(
                        structure, DB.data(), class_result)
                    if ligand_stats:
                        metal_stats.add_ligand(ligand_stats)
                        break
        else:
            ligand_stats = convalent_strategy.get_stats(
                structure, DB.data(), None)
            metal_stats.add_ligand(ligand_stats)

        if not metal_stats.is_empty():
            results.add_metal(metal_stats)

    Logger().info(
        f"Analysis completed. Statistics for {int(results.len())} ligands(metals) found.")
    return results
