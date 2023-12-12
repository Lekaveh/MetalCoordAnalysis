import os
import gemmi
import numpy as np
from tqdm import tqdm
from metalCoord.analysis.data import DB, PdbStats
from metalCoord.analysis.data import StrictCandidateFinder, ElementCandidateFinder, ElementInCandidateFinder, AnyElementCandidateFinder, NoCoordinationCandidateFinder
from metalCoord.analysis.data import StrictCorrespondenceStatsFinder, WeekCorrespondenceStatsFinder, OnlyDistanceStatsFinder
from metalCoord.analysis.structures import get_ligands
from metalCoord.load.rcsb import load_pdb
from metalCoord.logging import Logger






def get_structures(ligand, path):
  
    if os.path.isfile(path):
        st = gemmi.read_structure(path)
        return get_ligands(st, ligand)

    elif len(path) == 4:         
        pdb, type = load_pdb(path)
        if type == 'cif':
            cif_block = gemmi.cif.read_string(pdb)[0]
            st = gemmi.make_structure_from_block(cif_block)
        else:
            st = gemmi.read_pdb_string(pdb)
        return get_ligands(st, ligand)

    else:
        raise Exception("Existing pdb or mmcif file path should be provided or 4 letter pdb code")

    


strategies = [StrictCorrespondenceStatsFinder(StrictCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementInCandidateFinder()),
              WeekCorrespondenceStatsFinder(AnyElementCandidateFinder()),
              OnlyDistanceStatsFinder(NoCoordinationCandidateFinder())]


def find_classes(ligand, pdb_name):
    Logger().info(f"Analysing structres in  {pdb_name} for patterns")
    structures = get_structures(ligand, pdb_name)
    Logger().info(f"{len(structures)} structures found.")
    results = PdbStats()
    for structure in tqdm(structures, desc="Structures", position=0, disable=Logger().disabled):
        for strategy in tqdm(strategies, desc="Strategies", position=1, leave=False, disable=Logger().disabled):
            stats = strategy.get_stats(structure, DB.data())
            if not stats.isEmpty():
                results.addMetal(stats)
                break
    Logger().info(f"Analysis completed. Statistics for {results.len()} ligands(metals) found.")
    return results
