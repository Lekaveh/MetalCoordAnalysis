from load.rcsb import load_pdb
from analysis.structures import get_ligands
import gemmi
from gemmi import cif
import pandas as pd
from correspondense.procrustes import fit
import numpy as np
import argparse

data = pd.read_csv("./data/classes.zip")
distances = data.groupby(["Metal", "Ligand", "Class", "Coordination"]).Distance.agg(["mean", "std"]).reset_index()


def get_stats(ligand, pdb_name, output):
    pdb, type  = load_pdb(pdb_name)
    if type == 'cif':
        print("Unsupported data format cif")
    st = gemmi.read_pdb_string(pdb) 

    structures = get_ligands(st, ligand)
    result = []
    for structure in structures:
        selection = data[data.Code == structure.code()]
        classes = selection.Class.unique()
        files = [selection[selection.Class == cl].File.unique()[0] for cl in classes]
        
        for cl, file in zip(classes, files):
            file_data = selection[selection.File == file]
            coord1 = np.vstack([file_data[["MetalX", "MetalY", "MetalZ"]].values[:1], file_data[["LigandX", "LigandY", "LigandZ"]].values])
            coord2 = structure.get_coord()
            if fit(coord1, coord2)[0] < 0.3:
                metal = file_data["Metal"].values[0]
                ligands = file_data.loc[:, "Ligand"].unique()
                result.append(distances[(distances.Metal == metal) & (distances.Ligand.isin(ligands)) & (distances.Class == cl)])
    if len(result) > 0:    
        pd.concat(result, ignore_index=True).drop_duplicates().to_csv(output, index=False)
    else:
        print("No correspondense found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', required=True, help='ligand name')
    parser.add_argument('--pdb', required=True, help='pdb name')
    parser.add_argument('--output', required=True, help='output file path')

    args = parser.parse_args()

    ligand = args.ligand
    pdb_name = args.pdb
    output = args.output
    get_stats(ligand, pdb_name, output)
