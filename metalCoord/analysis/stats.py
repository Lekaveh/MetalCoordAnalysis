import os
import gemmi
import numpy as np
from tqdm import tqdm
from metalCoord.analysis.data import DB
from metalCoord.analysis.data import StrictCandidateFinder, ElementCandidateFinder, ElementInCandidateFinder, AnyElementCandidateFinder, NoCoordinationCandidateFinder
from metalCoord.analysis.data import StrictCorrespondenceStatsFinder, WeekCorrespondenceStatsFinder, OnlyDistanceStatsFinder
from metalCoord.analysis.structures import get_ligands
from metalCoord.load.rcsb import load_pdb
from metalCoord.logging import Logger

def ligandJSON(ligand):
    return {"name": ligand.name, "element": ligand.element, "chain": ligand.chain, "residue": ligand.residue, "sequence ": ligand.sequence}

def generateJson(stats):
    Logger().info(f"Generating json")
    results = []
    for s in stats:
        if s.isEmpty():
            continue
        metal = {"chain": s.chain, "residue": s.residue, "sequence ": s.sequence, "metal": s.metal,
                 "metalElement": s.metalElement, "ligands": [], "description": s.description}
        for l in s.ligands:
            clazz = {"class": l.clazz, "base": [], "angles": [], "pdb": [], "procrustes": str(
                np.round(l.procrustes, 3)), "coordination": l.coorditation,  "count": l.count}
            for b in l.base:
                clazz["base"].append(
                    {"ligand": ligandJSON(b.ligand), "distance": b.distance, "std": b.std})
            for a in l.angles:
                clazz["angles"].append({"ligand1": ligandJSON(a.ligand1),
                                        "ligand2": ligandJSON(a.ligand2),
                                        "angle": a.angle, "std": a.std})
            for p in l.pdb:
                clazz["pdb"].append({"ligand": ligandJSON(p.ligand), "distance": p.distance,
                                    "std": p.std })
            metal["ligands"].append(clazz)
        results.append(metal)
    Logger().info(f"Json generation completed")
    return results




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
    results = []
    for structure in tqdm(structures, desc="Structures", position=0, disable=Logger().disabled):
        for strategy in tqdm(strategies, desc="Strategies", position=1, leave=False, disable=Logger().disabled):
            stats = strategy.get_stats(structure, DB.data())
            if not stats.isEmpty():
                results.append(stats)
                break
    Logger().info(f"Analysis completed. {len(results)} results found.")
    return generateJson(results)
