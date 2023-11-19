import urllib.request
import urllib.error

def load_pdb(name):
    try:
        return (urllib.request.urlopen(f"https://files.rcsb.org/download/{name}.pdb").read(), "pdb")
    except urllib.error.HTTPError:
        return (urllib.request.urlopen(f"https://files.rcsb.org/download/{name}.cif").read(), "cif")

def load_ligand(name):
    return urllib.request.urlopen(f"https://files.rcsb.org/ligands/download/{name}.cif").read()