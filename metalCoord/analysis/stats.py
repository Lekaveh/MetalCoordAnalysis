
import numpy as np
from tqdm import tqdm

from metalCoord.load.rcsb import load_pdb
from metalCoord.analysis.structures import get_ligands
import gemmi
from metalCoord.analysis.data import DB
from metalCoord.analysis.data import StrictCandidateFinder, ElementCandidateFinder, ElementInCandidateFinder, AnyElementCandidateFinder, NoCoordinationCandidateFinder
from metalCoord.analysis.data import StrictCorrespondenceStatsFinder, WeekCorrespondenceStatsFinder, OnlyDistanceStatsFinder
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




def get_structures(ligand, pdb_name):
    if pdb_name.endswith(".pdb"):
        st = gemmi.read_pdb(pdb_name)   
    else:    
        pdb, type = load_pdb(pdb_name)
        if type == 'cif':
            Logger().error("Unsupported data format cif")
        st = gemmi.read_pdb_string(pdb)

    return get_ligands(st, ligand)



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
    for structure in tqdm(structures, desc="Structures", position=0):
        for strategy in tqdm(strategies, desc="Strategies", position=1, leave=False):
            stats = strategy.get_stats(structure, DB.data())
            if not stats.isEmpty():
                results.append(stats)
                break
    Logger().info(f"Analysis completed. {len(results)} results found.")
    return generateJson(results)
