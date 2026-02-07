import os
import math
from typing import Tuple
import gemmi
from tqdm import tqdm
from metalCoord.analysis.classes import Classificator
from metalCoord.analysis.data import DB
from metalCoord.analysis.cod import (
    StrictCandidateFinder,
    ElementCandidateFinder,
    ElementInCandidateFinder,
    AnyElementCandidateFinder,
    NoCoordinationCandidateFinder,
    CovalentCandidateFinder,
)
from metalCoord.analysis.stats import (
    OnlyDistanceStatsFinder,
    WeekCorrespondenceStatsFinder,
    StrictCorrespondenceStatsFinder,
    CovalentStatsFinder,
)
from metalCoord.analysis.metal import MetalPairStatsService
from metalCoord.analysis.models import MetalPairStats, MetalStats, PdbStats
from metalCoord.analysis.structures import get_ligands, get_ligands_from_cif, Ligand, MetalBondRegistry
from metalCoord.cif.utils import get_bonds, get_metal_metal_bonds
from metalCoord.load.rcsb import load_pdb
from metalCoord.logging import Logger
from metalCoord.config import Config



def read_structure(p: str) -> gemmi.Structure:
    """
    Read a macromolecular structure from disk or by 4-letter PDB accession code.

    This convenience function accepts either:
    1. A filesystem path pointing to an existing PDB (.pdb) or mmCIF (.cif/.mmcif) file.
    2. A 4-character PDB identifier (e.g. '1ABC'), in which case the structure is
        downloaded (via load_pdb) and parsed according to its format.

    Logic:
    - If p is an existing file path, it is parsed directly with gemmi.read_structure,
      which auto-detects PDB vs mmCIF.
    - If p has length 4, it is treated as a PDB code. The helper load_pdb returns
      the raw text plus a file_type indicator ("cif" or otherwise).
      mmCIF content is converted to a Structure via gemmi.make_structure_from_block;
      PDB content via gemmi.read_pdb_string.
    - Otherwise a FileNotFoundError is raised.

    Parameters
    ----------
    p : str
         Either a path to a local PDB/mmCIF file or a 4-letter PDB code.

    Returns
    -------
    gemmi.Structure
         Parsed structure object representing the molecular model.

    Raises
    ------
    FileNotFoundError
         If p is neither an existing file path nor a 4-character PDB code.

    Dependencies
    ------------
    gemmi : Used for reading and constructing Structure objects.
    load_pdb : External helper that retrieves PDB/mmCIF content and its format.

    Examples
    --------
    # From local file
    structure = read_structure("/data/structures/1abc.cif")

    # From PDB code
    structure = read_structure("1ABC")

    # Handling errors
    try:
         structure = read_structure("not_found.xyz")
    except FileNotFoundError as e:
         print(e)
    """
    if os.path.isfile(p):
        st = gemmi.read_structure(p)
    elif len(p) == 4:
        pdb_data, file_type = load_pdb(p)
        if file_type == "cif":
            cif_block = gemmi.cif.read_string(pdb_data)[0]
            st = gemmi.make_structure_from_block(cif_block)
        else:
            st = gemmi.read_pdb_string(pdb_data)
    else:
        raise FileNotFoundError(
            "Existing pdb or mmcif file path should be provided or 4 letter pdb code"
        )
    # st.ncs = []
    # st.setup_cell_images()
    return st


def get_structures(ligand, path, bonds=None, metal_metal_bonds=None, only_best=False) -> Tuple[list[Ligand], MetalBondRegistry]:
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

    st = read_structure(path)
    return get_ligands(st, ligand, bonds, metal_metal_bonds = metal_metal_bonds, only_best=only_best)


def _distance(atom1, atom2) -> float:
    return math.sqrt(
        (atom1.pos.x - atom2.pos.x) ** 2
        + (atom1.pos.y - atom2.pos.y) ** 2
        + (atom1.pos.z - atom2.pos.z) ** 2
    )


def _metal_site(structure: Ligand) -> dict:
    icode = structure.residue.seqid.icode.strip().replace("\x00", "")
    return {
        "metal": structure.metal.atom.name,
        "metalElement": str(structure.metal.element),
        "chain": structure.chain.name,
        "residue": structure.residue.name,
        "sequence": structure.residue.seqid.num,
        "icode": icode if icode else ".",
        "altloc": structure.metal.atom.altloc.strip().replace("\x00", ""),
        "symmetry": structure.metal.symmetry,
    }


def _ligand_environment(structure: Ligand) -> dict:
    metal_elem = structure.metal.atom.element.name
    base_ligands = list(structure.ligands)
    extra_ligands = list(structure.extra_ligands)
    ligands = []
    for atom in base_ligands + extra_ligands:
        cov_sum = gemmi.Element(metal_elem).covalent_r + gemmi.Element(atom.atom.element.name).covalent_r
        dist = _distance(structure.metal, atom)
        icode = atom.residue.seqid.icode.strip().replace("\x00", "")
        ligands.append(
            {
                "name": atom.atom.name,
                "element": atom.atom.element.name,
                "chain": atom.chain.name,
                "residue": atom.residue.name,
                "sequence": atom.residue.seqid.num,
                "icode": icode if icode else ".",
                "altloc": atom.atom.altloc.strip().replace("\x00", ""),
                "symmetry": atom.symmetry,
                "distance": round(dist, 3),
                "covalent_sum": round(cov_sum, 3),
                "covalent_ratio": round(dist / cov_sum, 3) if cov_sum else None,
                "is_extra": atom in extra_ligands,
            }
        )
    return {"ligands": ligands, "count": len(ligands)}

convalent_strategy = CovalentStatsFinder(CovalentCandidateFinder())
strategies = [
    StrictCorrespondenceStatsFinder(StrictCandidateFinder()),
    WeekCorrespondenceStatsFinder(ElementCandidateFinder()),
    WeekCorrespondenceStatsFinder(ElementInCandidateFinder()),
    WeekCorrespondenceStatsFinder(AnyElementCandidateFinder()),
    OnlyDistanceStatsFinder(NoCoordinationCandidateFinder()),
    convalent_strategy,
]


def find_classes_from_structures(
    structures: list[Ligand], bonds: dict, clazz: str = None
):
    """
    Analyzes a list of structures to classify them based on their coordination and other properties.

    Args:
        structures (list): A list of structure objects to be analyzed.
        bonds (int): The number of bonds to consider when cleaning the structures.

    Returns:
        PdbStats: An object containing the statistics of the analyzed structures.
    """

    debug_recorder = Config().debug_recorder if Config().debug else None

    for structure in tqdm(structures, disable=not Logger().progress_bars):
        Logger().info(
            f"Structure for {structure} found. Coordination number: {structure.coordination()}. {structure.name_code_with_symmetries()}"
        )
    Logger().info(f"{len(structures)} structures found.")
    results = PdbStats()
    classificator = Classificator()

    classes = {}
    for structure in tqdm(
        structures, desc="Structures", position=0, disable=not Logger().progress_bars
    ):
        structure_classes = []

        for class_result in tqdm(
            classificator.classify(structure, class_name=clazz),
            desc="Coordination",
            position=1,
            leave=False,
            disable=not Logger().progress_bars,
        ):
            structure_classes.append(class_result)

        if not structure_classes:
            new_structure = structure.clean_the_farthest(len(bonds))
            for class_result in tqdm(
                classificator.classify(new_structure, class_name=clazz),
                desc="Coordination",
                position=1,
                leave=False,
                disable=not Logger().progress_bars,
            ):
                structure_classes.append(class_result)
            if structure_classes:
                structure = new_structure
        classes[structure] = structure_classes

    for structure, structure_classes in classes.items():
        candidantes = []
        for class_result in structure_classes:
            candidantes.append(class_result.clazz)
        Logger().info(f"Candidates for {structure} : {candidantes}")

    for structure in tqdm(
        classes.keys(),
        desc="Structures",
        position=0,
        disable=not Logger().progress_bars,
    ):
        metal_stats = MetalStats(structure)
        strategy_map = {}
        if classes[structure]:
            for class_result in classes[structure]:
                chosen_strategy = None
                for strategy in tqdm(
                    strategies,
                    desc="Strategies",
                    position=1,
                    leave=False,
                    disable=not Logger().progress_bars,
                ):
                    ligand_stats = strategy.get_stats(
                        structure, DB.data(), class_result
                    )
                    if ligand_stats:
                        metal_stats.add_ligand(ligand_stats)
                        chosen_strategy = type(strategy).__name__
                        break
                if chosen_strategy:
                    strategy_map[class_result.clazz] = chosen_strategy
        else:
            ligand_stats = convalent_strategy.get_stats(structure, DB.data(), None)
            metal_stats.add_ligand(ligand_stats)
            strategy_map[""] = type(convalent_strategy).__name__

        if not metal_stats.is_empty():
            results.add_metal(metal_stats)

        if debug_recorder:
            candidates = []
            for class_result in classes[structure]:
                candidates.append(
                    {
                        "class": class_result.clazz,
                        "coordination": len(class_result.coord) - 1,
                        "procrustes": float(class_result.proc),
                    }
                )

            best = metal_stats.get_best_class() if not metal_stats.is_empty() else None
            chosen_class = best.clazz if best else None
            chosen_strategy = strategy_map.get(chosen_class, strategy_map.get("", None))
            debug_recorder.add_trace_structure(
                {
                    "metal_site": _metal_site(structure),
                    "ligand_environment": _ligand_environment(structure),
                    "coordination": {
                        "count": structure.coordination(),
                        "ligands": [lig.atom.name for lig in structure.ligands],
                        "extra_ligands": [lig.atom.name for lig in structure.extra_ligands],
                    },
                    "candidates": candidates,
                    "chosen_class": chosen_class,
                    "chosen_strategy": chosen_strategy,
                }
            )

    Logger().info(
        f"Analysis completed. Statistics for {int(results.len())} ligands(metals) found."
    )
    return results


def find_classes_pdb(
    ligand: str,
    pdb_name: str,
    bonds: dict = None,
    metal_metal_bonds: dict = None,
    only_best: bool = False,
    clazz: str = None,
) -> Tuple[PdbStats, list[MetalPairStats]]:
    """
    Analyzes structures in a given PDB file for patterns and returns the statistics for ligands (metals) found.

    Args:
        ligand (str): The name of the ligand.
        pdb_name (str): The name of the PDB file.
        bonds (dict, optional): A dictionary of bonds. Defaults to None.
        only_best (bool, optional): Flag indicating whether to consider only the best structures. Defaults to False.
        clazz (str, optional): The class to search for. Defaults to None.

    Returns:
        PdbStats: An object containing the statistics for ligands (metals) found.
    """
    if bonds is None:
        bonds = {}
    Logger().info(f"Analyzing structures in {pdb_name} for patterns")
    structures, metal_metal = get_structures(ligand, pdb_name, bonds, metal_metal_bonds, only_best)
    metal_pair_stats = MetalPairStatsService().get_metal_pair_stats(metal_metal)

    return find_classes_from_structures(structures, bonds, clazz=clazz), metal_pair_stats


def find_classes_cif(
    name: str, atoms: gemmi.cif.Table, bonds: gemmi.cif.Table, clazz: str = None
) -> Tuple[PdbStats, MetalPairStats]:
    """Classify ligands and bonds from CIF data.

    This function takes CIF tables of atoms and bonds, processes them to extract
    ligands and bonds, and then classifies them into different categories.

    Args:
        name (str): The name of the ligand.
        atoms (gemmi.cif.Table): A CIF table containing atomic data.
        bonds (gemmi.cif.Table): A CIF table containing bond data.
        clazz (str, optional): The class to search for. Defaults to None.

    Returns:
        PdbStats: An object containing statistics and classifications of the ligands and bonds.
    """
    b = get_bonds(atoms, bonds)
    m_b = get_metal_metal_bonds(atoms, bonds)
    ligands, metal_metal_bonds = get_ligands_from_cif(name, atoms, b, m_b)
    metal_pair_stats = MetalPairStatsService().get_metal_pair_stats(metal_metal_bonds)
    return find_classes_from_structures(
        ligands, b, clazz=clazz
    ), metal_pair_stats
