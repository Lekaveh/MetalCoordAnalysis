
import numpy as np
from tqdm import tqdm

from metalCoord.load.rcsb import load_pdb
from metalCoord.analysis.structures import get_ligands
import gemmi
from metalCoord.analysis.data import DB
from metalCoord.analysis.data import StrictCandidateFinder, ElementCandidateFinder, ElementInCandidateFinder, AnyElementCandidateFinder, NoCoordinationCandidateFinder
from metalCoord.analysis.data import StrictCorrespondenceStatsFinder, WeekCorrespondenceStatsFinder, OnlyDistanceStatsFinder


def ligandJSON(ligand):
    return {"name": ligand.name, "element": ligand.element, "chain": ligand.chain, "residue": ligand.residue, "sequence ": ligand.sequence}

def generateJson(stats):
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
    return results




def get_structures(ligand, pdb_name):
    pdb, type = load_pdb(pdb_name)
    if type == 'cif':
        print("Unsupported data format cif")
    st = gemmi.read_pdb_string(pdb)

    return get_ligands(st, ligand)



strategies = [StrictCorrespondenceStatsFinder(StrictCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementInCandidateFinder()),
              WeekCorrespondenceStatsFinder(AnyElementCandidateFinder()),
              OnlyDistanceStatsFinder(NoCoordinationCandidateFinder())]


def find_classes(ligand, pdb_name):
    structures = get_structures(ligand, pdb_name)
    results = []
    for structure in tqdm(structures):
        for strategy in strategies:
            stats = strategy.get_stats(structure, DB.data())
            if not stats.isEmpty():
                results.append(stats)
                break
    return generateJson(results)
