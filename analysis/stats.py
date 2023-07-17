
import pandas as pd
from correspondense.procrustes import fit
import numpy as np
from load.rcsb import load_pdb
from analysis.structures import get_ligands
import gemmi


data = pd.read_csv("./data/classes.zip")
distances = data.groupby(["Metal", "Ligand", "Class", "Coordination"]).Distance.agg(["mean", "std"]).reset_index()

def get_distance_std(metal, ligand, cl):
    return distances[(distances.Metal == metal) & (distances.Ligand == ligand) & (distances.Class == cl)][["mean", "std"]].values[0]


def get_structures(ligand, pdb_name):
    pdb, type  = load_pdb(pdb_name)
    if type == 'cif':
        print("Unsupported data format cif")
    st = gemmi.read_pdb_string(pdb) 

    return get_ligands(st, ligand)

def find_classes(ligand, pdb_name):


    structures = get_structures(ligand, pdb_name)
    results = []
    for structure in structures:
        selection = data[data.Code == structure.code()]
        classes = selection.Class.unique()
        files = [selection[selection.Class == cl].File.unique()[0] for cl in classes]
        result = {"chain": structure.chain.name, "residue":structure.residue.name, "sequence ":structure.residue.seqid.num, "metal":structure.metal.name, "metalElement":structure.metal.element.name, "ligands":[]}
        for cl, file in zip(classes, files):
            file_data = selection[selection.File == file]
            coord1 = np.vstack([file_data[["MetalX", "MetalY", "MetalZ"]].values[:1], file_data[["LigandX", "LigandY", "LigandZ"]].values])
            coord2 = structure.get_coord()
            proc_dist = fit(coord1, coord2)[0]
            if proc_dist < 0.3:
      
                clazz = {"class":cl, "base":[], "procrustes":str(np.round(proc_dist, 3)), "coordination":structure.coordination()}
                for l in structure.ligands:
                    dist, std = get_distance_std(structure.metal.element.name, l.element.name, cl)

                    clazz["base"].append({"ligand":l.name, "ligandElement":l.element.name, "distance":dist, "std":std})
                if len(structure.extra_ligands) > 0:
                    clazz["pdb"] = []
                    for l in structure.extra_ligands:
                        dist, std = get_distance_std(structure.metal.element.name, l.atom.element.name, cl)
                        clazz["pdb"].append({"ligand":l.atom.name, "ligandElement":l.atom.element.name, "distance":dist, "std":std, "chain": l.chain.name, "residue":l.residue.name, "sequence ":l.residue.seqid.num,})    
                result["ligands"].append(clazz)
        if len(result["ligands"]) > 0:
            results.append(result)
    return results