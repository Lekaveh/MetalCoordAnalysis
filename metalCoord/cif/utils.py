import gemmi
# Constants
ANGLE_CATEGORY = "_chem_comp_angle"
BOND_CATEGORY = "_chem_comp_bond"
ATOM_CATEGORY = "_chem_comp_atom"
COMP_CATEGORY = "_chem_comp"
ACEDRG_CATEGORY = "_acedrg_chem_comp_descriptor"

COMP_ID = "comp_id"
ATOM_ID = "atom_id"
TYPE_SYMBOL = "type_symbol"
ENERGY = "type_energy"
ATOM_ID_1 = "atom_id_1"
ATOM_ID_2 = "atom_id_2"
ATOM_ID_3 = "atom_id_3"
VALUE_DIST_NUCLEUS = "value_dist_nucleus"
VALUE_DIST_NUCLEUS_ESD = "value_dist_nucleus_esd"
VALUE_DIST = "value_dist"
VALUE_DIST_ESD = "value_dist_esd"
VALUE_ANGLE = "value_angle"
VALUE_ANGLE_ESD = "value_angle_esd"
ID = "id"
NAME = "name"
GROUP = "group"
NUMBER_ATOMS_ALL = "number_atoms_all"
NUMBER_ATOMS_NH = "number_atoms_nh"
DESC_LEVEL = "desc_level"
THREE_LETTER_CODE = "three_letter_code"
PROGRAM_NAME = "program_name"
PROGRAM_VERSION = "program_version"
TYPE = "type"
COORDS = [[f"pdbx_model_Cartn_{sym}_ideal",
           f"model_Cartn_{sym}", f"{sym}"] for sym in ["x", "y", "z"]]
COORDS = list(map(list, zip(*COORDS)))



def get_element_name(mmcif_atom_category, name):
    """
    Get the element name for a given atom name.

    Parameters:
    - mmcif_atom_category (dict): The mmcif atom category.
    - name (str): The name of the atom.

    Returns:
    - element_name (str): The element name of the atom.
    """
    for i, _ in enumerate(mmcif_atom_category[ATOM_ID]):
        if _ == name:
            return gemmi.Element(mmcif_atom_category[TYPE_SYMBOL][i]).name
        

def get_bonds(atoms: gemmi.cif.Table, bonds: gemmi.cif.Table) -> dict:
    """
    Identify and return bonds involving metal atoms from the given atoms and bonds data.

    Args:
        atoms (list): A list of atom data.
        bonds (dict): A dictionary containing bond information with keys ATOM_ID_1 and ATOM_ID_2.

    Returns:
        dict: A dictionary where keys are metal atom IDs and values are lists of atom IDs bonded to the metal atom.

    Notes:
        - Bonds between two non-metal atoms are ignored.
        - Bonds between two metal atoms are ignored.
        - If a bond involves a metal atom and a non-metal atom, the metal atom is used as the key in the result dictionary.
    """
    if not bonds:
        return {}
    result = {}
    for atom1, atom2 in zip(bonds[ATOM_ID_1], bonds[ATOM_ID_2]):
        if not gemmi.Element(get_element_name(atoms, atom1)).is_metal and not gemmi.Element(get_element_name(atoms, atom2)).is_metal:
            continue
        if gemmi.Element(get_element_name(atoms, atom1)).is_metal and gemmi.Element(get_element_name(atoms, atom2)).is_metal:
            continue

        if gemmi.Element(get_element_name(atoms, atom2)).is_metal:
            atom1, atom2 = atom2, atom1

        result.setdefault(atom1, []).append(atom2)

    return result