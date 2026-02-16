import os
from pathlib import Path
import urllib.error
import urllib.request


def _read_local_pdb_file(name: str):
    name = name.lower()
    candidates = [f"{name}.pdb", f"{name}.cif", f"{name}.mmcif"]
    search_dirs = []
    cache_dir = os.getenv("METALCOORD_PDB_CACHE")
    if cache_dir:
        search_dirs.append(Path(cache_dir))

    for parent in Path(__file__).resolve().parents:
        models_dir = parent / "tests" / "data" / "models"
        if models_dir.is_dir():
            search_dirs.append(models_dir)
            break

    for directory in search_dirs:
        for candidate in candidates:
            candidate_path = directory / candidate
            if candidate_path.is_file():
                data = candidate_path.read_bytes()
                file_type = "pdb" if candidate_path.suffix.lower() == ".pdb" else "cif"
                return data, file_type
    return None


def load_pdb(name):
    local_data = _read_local_pdb_file(name)
    if local_data:
        return local_data
    try:
        return (urllib.request.urlopen(f"https://files.rcsb.org/download/{name}.pdb").read(), "pdb")
    except urllib.error.HTTPError:
        return (urllib.request.urlopen(f"https://files.rcsb.org/download/{name}.cif").read(), "cif")


def load_ligand(name):
    return urllib.request.urlopen(f"https://files.rcsb.org/ligands/download/{name}.cif").read()
