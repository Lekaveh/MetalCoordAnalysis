
import pandas as pd
from ..correspondense.procrustes import fit
import numpy as np
from ..load.rcsb import load_pdb
from .structures import get_ligands
import gemmi
import os
import sys
import json

d = os.path.dirname(sys.modules["metalCoord"].__file__)

data = pd.read_csv(os.path.join(d, "data/classes.zip"))

distances = data.groupby(["Metal", "Ligand", "Class", "Coordination"]).Distance.agg(["mean", "std"]).reset_index()




def get_distance_std(metal, ligand, cl):
    return distances[(distances.Metal == metal) & (distances.Ligand == ligand) & (distances.Class == cl)][["mean", "std"]].values[0]


def get_structures(ligand, pdb_name):
    pdb, type  = load_pdb(pdb_name)
    if type == 'cif':
        print("Unsupported data format cif")
    st = gemmi.read_pdb_string(pdb) 

    return get_ligands(st, ligand)

def get_groups(atoms1, atoms2):
    unique_atoms = np.unique(atoms1)
    group1 = []
    group2 = []
    for atom in unique_atoms:
        group1.append(np.where(atoms1 == atom)[0].tolist())
        group2.append(np.where(atoms2 == atom)[0].tolist())
    
    return [group1, group2]

def angle(metal, ligand1, ligand2):
    a = metal - ligand1
    b = metal - ligand2
    a = np.array(a)/np.linalg.norm(a)
    b = np.array(b)/np.linalg.norm(b)
    cosine_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.rad2deg (np.arccos(cosine_angle))

def find_classes(ligand, pdb_name, with_angles=True):


    structures = get_structures(ligand, pdb_name)
    results = []
    for structure in structures:
        metal_name = structure.metal.element.name
        selection = data[data.Code == structure.code()]
        classes = selection.Class.unique()
        class_files = [selection[selection.Class == cl].File.unique() for cl in classes]
        result = {"chain": structure.chain.name, "residue":structure.residue.name, "sequence ":structure.residue.seqid.num, "metal":structure.metal.name, "metalElement":structure.metal.element.name, "ligands":[]}
        o_ligand_atoms = np.array([metal_name] + structure.atoms())
        o_ligand_coord = structure.get_coord()
        
        for cl, files in zip(classes, class_files):
            distances = []
            angles = []
            for file in files:
                file_data = selection[selection.File == file]
                m_ligand_coord = np.vstack([file_data[["MetalX", "MetalY", "MetalZ"]].values[:1], file_data[["LigandX", "LigandY", "LigandZ"]].values])
               
                m_ligand_atoms = np.insert(file_data[["Ligand"]].values.ravel(), 0, metal_name)
                groups = get_groups(o_ligand_atoms,  m_ligand_atoms)
                proc_dists, indices, min_proc_dist = fit(o_ligand_coord, m_ligand_coord, groups=groups, all=True)
                if min_proc_dist < 0.3:
                    for proc_dist, index in zip(proc_dists,indices):
                        distances.append(np.sqrt(np.sum((m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis = 1))[1:].tolist())
                        if with_angles:
                            angles.append([angle(m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j]) for i in range(1, len(o_ligand_coord) - 1) for j in range(i + 1, len(o_ligand_coord))])
                               

            distances = np.array(distances).T
            angles = np.array(angles).T
            if (distances.shape[0] > 0):
                clazz = {"class":cl, "base":[], "procrustes":str(np.round(min_proc_dist, 3)), "coordination":structure.coordination(),  "count": distances.shape[1]}
                
                n_ligands = len(structure.ligands)
                ligands = structure.ligands
                for i, l in enumerate(ligands):
                    dist, std = distances[i].mean(), distances[i].std()
                    clazz["base"].append({"ligand":l.name, "ligandElement":l.element.name, "distance":dist, "std":std})
                if with_angles:
                    k = 0
                    clazz["angles"] = []
                    for i in range(n_ligands - 1):    
                        for j in range(i + 1, n_ligands):
                            a, std = angles[k].mean(), angles[k].std() 
                            clazz["angles"].append({"ligand1":ligands[i].name, "ligandElement1":ligands[i].element.name, 
                                                    "ligand2":ligands[j].name, "ligandElement2":ligands[j].element.name, 
                                                    "angle":a, "std":std})
                            k += 1
                       

                if len(structure.extra_ligands) > 0:
                    clazz["pdb"] = []
                    for i, l in enumerate(structure.extra_ligands):
                        dist, std = distances[i + n_ligands].mean(), distances[i + n_ligands].std()
                        clazz["pdb"].append({"ligand":l.atom.name, "ligandElement":l.atom.element.name, "distance":dist, "std":std, "chain": l.chain.name, "residue":l.residue.name, "sequence ":l.residue.seqid.num,})    
                result["ligands"].append(clazz)

        if len(result["ligands"]) > 0:
            results.append(result)
    return results