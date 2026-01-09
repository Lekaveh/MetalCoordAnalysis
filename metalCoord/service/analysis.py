from dataclasses import dataclass
from collections import defaultdict
import re
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, Union, Callable

import gemmi
import networkx as nx
import metalCoord
from metalCoord.analysis.classify import find_classes_pdb, find_classes_cif, read_structure

from metalCoord.analysis.metal import MetalPairStatsService
from metalCoord.analysis.models import DistanceStats, LigandStats, PdbStats, MetalStats
from metalCoord.config import Config
from metalCoord.logging import Logger
from metalCoord.cif.utils import (
    ACEDRG_CATEGORY,
    ATOM_CATEGORY,
    BOND_CATEGORY,
    ANGLE_CATEGORY,
    COMP_CATEGORY,
    COMP_ID,
    PROGRAM_NAME,
    PROGRAM_VERSION,
    TYPE,
    ATOM_ID,
    TYPE_SYMBOL,
    VALUE_DIST,
    VALUE_DIST_ESD,
    VALUE_DIST_NUCLEUS,
    VALUE_DIST_NUCLEUS_ESD,
    VALUE_ANGLE,
    VALUE_ANGLE_ESD,
    ATOM_ID_1,
    ATOM_ID_2,
    ATOM_ID_3,
    ID,
    THREE_LETTER_CODE,
    NAME,
    GROUP,
    NUMBER_ATOMS_ALL,
    NUMBER_ATOMS_NH,
    DESC_LEVEL,
    ENERGY,
)
from metalCoord.cif.utils import get_element_name, get_bonds, get_metal_metal_bonds



d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json"), encoding="utf-8"))


def decompose(values, n):
    """
    Decompose a list of values into sublists of size n.

    Args:
        values (list): The list of values to be decomposed.
        n (int): The size of each sublist.

    Returns:
        list: A list of sublists, each containing n values.
    """
    result = []
    for i in range(0, len(values) // n):
        result.append(values[i * n : i * n + n])
    return result


def pack(values):
    """
    Pack a list of values into sublists, grouping them by index.

    Args:
        values (list): The list of values to be packed.

    Returns:
        list: A list of sublists, each containing the values at the same index.
    """
    return [list(x) for x in zip(*values)]


def get_distance(clazz, ligand_name):
    """
    Get the distance, standard deviation, class, procrustes value, 
    and coordination
    for a given ligand name in a coordination class.

    Parameters:
    - clazz (dict): The coordination class containing ligand information.
    - ligand_name (str): The name of the ligand to retrieve information for.

    Returns:
    - distance (float): The distance of the ligand.
    - std (float): The standard deviation of the ligand distance.
    - cl (str): The class of the coordination.
    - procrustes (float): The procrustes value of the coordination.
    - coordination (int): The coordination number of the coordination.

    If clazz is None, all return values will be set to their default values.
    """
    coordination = 0
    cl = ""
    procrustes = 1
    distance = -1
    std = -1

    if clazz is None:
        return distance, std, cl, procrustes, coordination

    for ligand in clazz["base"]:
        if ligand["ligand"]["name"] == ligand_name:
            coordination = clazz["coordination"]
            procrustes = float(clazz["procrustes"])
            distance = ligand["distance"]
            std = ligand["std"]
            cl = clazz["class"]
            break

    return distance, std, cl, procrustes, coordination


def get_angles(clazz, ligand_name1, ligand_name2):
    """
    Get the angle, standard deviation, class, procrustes value, 
    and coordination
    for a given pair of ligand names in a coordination class.

    Parameters:
    - clazz (dict): The coordination class containing ligand information.
    - ligand_name1 (str): The name of the first ligand.
    - ligand_name2 (str): The name of the second ligand.

    Returns:
    - angle (float): The angle between the ligands.
    - std (float): The standard deviation of the angle.
    - cl (str): The class of the coordination.
    - procrustes (float): The procrustes value of the coordination.
    - coordination (int): The coordination number of the coordination.

    If clazz is None, all return values will be set to their default values.
    """
    coordination = 0
    cl = ""
    procrustes = 1
    angle = -1
    std = -1

    if clazz is None:
        return angle, std, cl, procrustes, coordination

    for ligand in clazz["angles"]:
        if (
            (ligand["ligand1"]["name"] == ligand_name1)
            and (ligand["ligand2"]["name"] == ligand_name2)
        ) or (
            (ligand["ligand2"]["name"] == ligand_name1)
            and (ligand["ligand1"]["name"] == ligand_name2)
        ):
            coordination = clazz["coordination"]
            procrustes = float(clazz["procrustes"])
            angle = ligand["angle"]
            std = ligand["std"]
            cl = clazz["class"]
            break

    return angle, std, cl, procrustes, coordination


def code(ligand1_name, metal_name, ligand2_name):
    """
    Generate a code by sorting and concatenating the ligand names.

    Parameters:
    - ligand1_name (str): The name of the first ligand.
    - metal_name (str): The name of the metal.
    - ligand2_name (str): The name of the second ligand.

    Returns:
    - code (str): The generated code.
    """
    return "".join(sorted([ligand1_name, metal_name, ligand2_name]))


def contains_metal(mmcif_atom_category):
    """
    Check if the mmcif atom category contains any metal atoms.

    Parameters:
    - mmcif_atom_category (dict): The mmcif atom category.

    Returns:
    - contains_metal (bool): True if the category contains metal atoms,
      False otherwise.
    """
    for element in mmcif_atom_category[TYPE_SYMBOL]:
        if gemmi.Element(element).is_metal:
            return True
    return False


def find_minimal_cycles(bonds):
    """
    Find the minimal cycles in a graph represented by a list of bonds.

    Parameters:
    - bonds (dict): The dictionary containing the atom IDs of the bonds.

    Returns:
    - cycle_basis (list): A list of minimal cycles in the graph.
    """
    graph = nx.Graph()
    graph.add_edges_from([[a1, a2] for a1, a2 in bonds])
    # Find the cycle basis of the graph
    cycle_basis = nx.simple_cycles(graph, length_bound=4)

    return cycle_basis


def bond_exist(bonds, atom1, atom2):
    """
    Check if a bond exists between two atoms.

    Parameters:
    - bonds (dict): The dictionary containing the atom IDs of the bonds.
    - atom1 (str): The ID of the first atom.
    - atom2 (str): The ID of the second atom.

    Returns:
    - bond_exists (bool): True if the bond exists, False otherwise.
    """
    for b1, b2 in zip(bonds[ATOM_ID_1], bonds[ATOM_ID_2]):
        if (b1 == atom1 and b2 == atom2) or (b1 == atom2 and b2 == atom1):
            return True
    return False


def save_cods(pdb_stats: PdbStats, path: str):
    """
    Save COD files for ligands in the given PdbStats object.

    Args:
        pdb_stats (PdbStats): The PdbStats object containing the ligand 
            information.
        path (str): The path where the COD files will be saved.

    Returns:
        None
    """
    output = os.path.join(path, "cod")
    for monomer in pdb_stats.monomers():
        for metal_stats in monomer.metals:
            for ligand_stats in metal_stats.ligands:
                output_folder = os.path.join(
                    output,
                    metal_stats.residue,
                    metal_stats.chain,
                    str(metal_stats.sequence),
                    metal_stats.metal,
                    ligand_stats.clazz,
                )
                Path(output_folder).mkdir(exist_ok=True, parents=True)
                for file, st in ligand_stats.cods:
                    st.write_pdb(os.path.join(output_folder, file))

def extract_name_from_doc(doc: gemmi.cif.Document) -> str:
    """
    Extracts the ligand name from a gemmi.cif.Document containing multiple blocks.

    This function iterates over each block in the provided document and uses a regular expression
    to search for a valid ligand name in the format "<name>" or "comp_<name>", where <name> consists
    of at least three alphanumeric characters. The function ignores names that are exactly "list".
    If a matching ligand name is found, it logs the name and returns it. If no valid ligand name
    is found in any block, the function raises a ValueError.

    Parameters:
        doc (gemmi.cif.Document): A document composed of multiple blocks, each with a 'name' attribute.

    Returns:
        str: The extracted ligand name.

    Raises:
        ValueError: If no block with a valid ligand name is found.
    """

    for block in doc:
        matches = re.findall(r"^(?:comp_)?([A-Za-z0-9]{3,}$)", block.name)
        if matches:
            name = matches[0]
            if name == "list":
                continue
            Logger().info(f"Ligand {name} found")
            return name
    raise ValueError("No block found for <name>|comp_<name>. "
                     "Please check the CIF file.")


def extract_block_from_doc(name: str, doc: gemmi.cif.Document) -> gemmi.cif.Block:
    """
    Extracts a block from the document based on the given name.
    The function first attempts to locate a block in the provided document using the identifier
    "comp_{name}". If this search returns None, it then searches using the plain "{name}" identifier.
    If no block is found through either method, a ValueError is raised to indicate the absence
    of the expected block in the document.
    Args:
        name (str): The identifier used to locate the desired block.
        doc (object): An object representing the document which must implement the 'find_block'
                      method to search for blocks.
    Returns:
        object (gemmi.cif.Block): The block from the document that matches the given identifier.
    Raises:
        ValueError: If no block corresponding to either "comp_{name}" or "{name}" is found in the document.
    """
    block = (
        doc.find_block(f"comp_{name}")
        if doc.find_block(f"comp_{name}") is not None
        else doc.find_block(f"{name}")
    )

    if block is None:
        raise ValueError(
            f"No block found for {name}|comp_{name}. "
            "Please check the CIF file."
        )
        
    return block

def update_ace_drg(block: gemmi.cif.Block, name: str):
    """
    Update the ACE/DRG 
    category of a CIF block 
    with metal coordination 
    analysis details.

    This function retrieves 
    the ACE/DRG category 
    from the provided CIF block. 
    If the category does not 
    exist, it creates a new one 
    with predefined keys 
    (COMP_ID, PROGRAM_NAME, 
    PROGRAM_VERSION, TYPE). 
    It then appends the 
    provided component name 
    along with the analysis 
    program details (name, 
    version, and description) 
    to the respective fields 
    of the ACE/DRG category. 
    Finally, the updated 
    category is set back 
    into the block.

    Parameters:
        block (gemmi.cif.Block): 
            The CIF block to update.
        name (str): 
            The component identifier 
            to add to the ACE/DRG category.

    Side Effects:
        Modifies the provided CIF block 
        by updating or creating 
        its ACE/DRG category.
    """
    ace_drg = block.get_mmcif_category(ACEDRG_CATEGORY)
    if not ace_drg:
        ace_drg = {
            COMP_ID: [],
            PROGRAM_NAME: [],
            PROGRAM_VERSION: [],
            TYPE: []
        }
    ace_drg[COMP_ID].append(name)
    ace_drg[PROGRAM_NAME].append("metalCoord")
    ace_drg[PROGRAM_VERSION].append(metalCoord.__version__)
    ace_drg[TYPE].append("metal coordination analysis")

    block.set_mmcif_category(ACEDRG_CATEGORY, ace_drg)

# Get the mmcif categories
def get_atoms_category(block: gemmi.cif.Block) -> Dict[str, Any]:
    """
    Retrieve the atoms category from a given mmCIF block.
    
    This function extracts the atoms category,
    which is defined by the constant
    `ATOM_CATEGORY`, from a given
    `gemmi.cif.Block`.
    If the category is not found,
    this function raises a ValueError.
    The error indicates that the CIF file
    may be missing required data.
    
    Parameters:
        block (gemmi.cif.Block):
            The CIF block that contains the data
            to extract the atoms category information.
    
    Returns:
        Dict[str, Any]:
            A dictionary that holds the atoms category
            as extracted from the CIF block.
    
    Raises:
        ValueError:
            If the atoms category (`ATOM_CATEGORY`)
            is not present in the provided block.
    """
    atoms = block.get_mmcif_category(ATOM_CATEGORY)
    if not atoms:
        raise ValueError(
            f"mmcif category {ATOM_CATEGORY} not found. Please check the CIF file."
        )
    return atoms

def update_energy_in_atoms(block: gemmi.cif.Block, atoms: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the energy values in the atoms dictionary within a CIF block.

    This function checks whether the ENERGY field exists in the provided atoms dictionary.
    If the ENERGY field is missing, it creates a copy of the dictionary, copies the value from the
    TYPE_SYMBOL field to the ENERGY field as a default, and then updates the corresponding category
    in the CIF block using block.set_mmcif_category with ATOM_CATEGORY. The function returns the new
    atoms dictionary with the updated ENERGY field. If the ENERGY field already exists, the original
    atoms dictionary is returned unmodified.

    Parameters:
        block (gemmi.cif.Block): A CIF block object that is used to update a specific category with the new atoms data.
        atoms (Dict[str, Any]): A dictionary containing atom-related data, expected to include at least the
                                TYPE_SYMBOL key, and optionally the ENERGY key.

    Returns:
        Dict[str, Any]: The updated atoms dictionary containing the ENERGY field. If ENERGY was absent,
                        it is created by copying the value from TYPE_SYMBOL; otherwise, the original dictionary is returned.

    Side Effects:
        Updates the specified category in the provided CIF block via block.set_mmcif_category when the ENERGY
        field is added.
    """
    if ENERGY not in atoms:
        new_atoms = {key: value for key, value in atoms.items()}
        # Copy the TYPE_SYMBOL values to the ENERGY field as default
        new_atoms[ENERGY] = atoms[TYPE_SYMBOL]
        block.set_mmcif_category(ATOM_CATEGORY, new_atoms)
        return new_atoms
    return atoms

#################################################################################################################



@dataclass
class RingStats:
    ring_type: Optional[str]
    coordination: int
    other_coordination: int
    class_name: str
    other_class_name: str
    metal_name: str
    other_metal_name: str
    

class AngleStatClassifier:
    def __init__(
        self,
        condition: Optional[Callable[[RingStats], Any]] = None,
        statistics: Optional[Dict[Any, "AngleStatClassifier"]] = None,
        result: Optional[Union[Callable[[RingStats], Dict], Dict]] = None
    ):
        self.condition = condition
        self.statistics = statistics or {}
        self.result = result

    def get_angles_stats(self, ring_stats: RingStats) -> Dict:
        if self.result is not None:
            return self.result(ring_stats) if callable(self.result) else self.result
        if self.condition is None:
            raise ValueError("No condition defined for classifier node")
        key = self.condition(ring_stats)
        child = self.statistics.get(key)
        if child is None:
            return {}  
        return child.get_angles_stats(ring_stats)


def build_angle_classifier() -> AngleStatClassifier:
    return AngleStatClassifier(
        condition=lambda r: r.ring_type if r.ring_type is not None else "other",
        statistics={
            "s-fe-s-fe": AngleStatClassifier(
                condition=lambda r: (
                    (r.coordination, r.other_coordination)
                    if r.coordination == r.other_coordination and r.coordination in {4, 5, 6, 8}
                    else None
                ),
                statistics={
                    (4, 4): AngleStatClassifier(
                        condition=lambda r: (
                            (r.class_name, r.other_class_name)
                            if r.class_name == 'tetrahedral' and r.other_class_name == 'tetrahedral'  # to improve
                            else None
                        ),
                        statistics={
                            ('tetrahedral', 'tetrahedral'): AngleStatClassifier(
                                result=lambda r: { # most of 4 coord metals in ring have these stats
                                    r.metal_name: {'angle': 104.89, 'std': 4.13},
                                    r.other_metal_name: {'angle': 104.89, 'std': 4.13},
                                    'S': {'angle': 74.47, 'std': 4.05},
                                }
                            ),
                            None: AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
                        }
                    ),
                    (5, 5): AngleStatClassifier(
                        condition=lambda r: (
                            (r.class_name, r.other_class_name)
                            if r.class_name == 'square-pyramid' and r.other_class_name == 'square-pyramid'  # to improve
                            else None
                        ),
                        statistics={
                            ('square-pyramid', 'square-pyramid'): AngleStatClassifier(
                                result=lambda r: { # most of 5 coord metals in ring have these stats
                                    r.metal_name: {'angle': 84.77, 'std': 1.10},
                                    r.other_metal_name: {'angle': 84.77, 'std': 1.10},
                                    'S': {'angle': 68.06, 'std': 1.18}, 
                                }
                            ),
                            None: AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
                        }
                    ),
                    (6, 6): AngleStatClassifier(
                        condition=lambda r: (
                            (r.class_name, r.other_class_name)
                            if r.class_name == 'octahedral' and r.other_class_name == 'octahedral'  # to improve
                            else None
                        ),
                        statistics={
                            ('octahedral', 'octahedral'): AngleStatClassifier(
                                result=lambda r: {  # most of 6 coord metals in ring have these stats
                                    r.metal_name: {'angle': 82.35, 'std': 2.56},
                                    r.other_metal_name: {'angle': 82.35, 'std': 2.56},
                                    'S': {'angle': 95.47, 'std': 3.25}, # hardcoded, there is geometrical differences
                                }
                            ),
                            None: AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
                        }
                    ),
                    (8, 8): AngleStatClassifier(result=lambda r: {}),  # ambiguities
                    None: AngleStatClassifier(result=lambda r: {}),   # unknown / add more variations
                }
            ),
            "s-fe-s-m": AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
            "s-fe-s-nm": AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
            "other": AngleStatClassifier(result=lambda r: {}), # unknown / add more variations
        }
    )

CLASSIFIER = build_angle_classifier()
METALS = {gemmi.Element(i).name.upper() for i in range(1, 119) if gemmi.Element(i).is_metal}


def get_ring_angles_stats(metal_stats1: MetalStats, ligand_stats1: LigandStats,
              metal_stats2: MetalStats, ligand_stats2: LigandStats) -> Dict[str, Dict[str, float]]:
    """
    Estimate approximate bond angles and their variance for a ring.

    This function uses a decision-tree-like classifier to predict angles based on the
    coordination numbers, coordination geometry classes, and types of two metal atoms
    connected by shared ligands in a ring.

    The function dynamically computes the angles using a nested `AngleStatClassifier`,
    which supports multi-level conditional decisions.

    Parameters:
        metal_stats1 (Atom): The first metal atom object in the ring.
        ligand_stats1 (LigandStats): Coordination and geometry information for metal_stats1.
        metal_stats2 (Atom): The second metal atom object in the ring.
        ligand_stats2 (LigandStats): Coordination and geometry information for metal_stats2.

    Returns:
        dict: A dictionary mapping metal atom names to their estimated angles and std.
        If the classifier does not have statistics for a given combination, an empty dict is returned.

    Raises:
        TypeError: If any of the inputs are not instances of Atom or LigandStats.
        ValueError: If the provided atoms are not metals or if the coordination numbers are missing.
        KeyError: If the classifier encounters a combination of properties that is not covered by the decision tree.
    """

    if not isinstance(metal_stats1, MetalStats):
        raise TypeError(f"Expected MetalStats for metal_stats1, got {type(metal_stats1)}")
    if not isinstance(ligand_stats1, LigandStats):
        raise TypeError(f"Expected LigandStats for ligand_stats1, got {type(ligand_stats1)}")
    if not isinstance(metal_stats2, MetalStats):
        raise TypeError(f"Expected MetalStats for metal_stats1, got {type(metal_stats2)}")
    if not isinstance(ligand_stats2, LigandStats):
        raise TypeError(f"Expected LigandStats for ligand_stats2, got {type(ligand_stats2)}")

    metal1_name, metal1_element, _, _, _ = metal_stats1.code
    metal2_name, metal2_element, _, _, _ = metal_stats2.code

    if metal1_element.upper() not in METALS or metal2_element.upper() not in METALS:
        raise ValueError("Provided metal_atoms are not metals")

    if ligand_stats1.coordination is None:
        raise ValueError(f"Coordination missing for {metal1_name}")
    if ligand_stats2.coordination is None:
        raise ValueError(f"Coordination missing for {metal2_name}")

    # need to check for simpicity of the structures

    atoms_in_ligand2 = (
        [d.ligand for d in ligand_stats2.bonds] +
        [d.ligand for d in ligand_stats2.pdb]
    )

    shared_atoms = [ # using equals methods for finding shared atoms
        d.ligand
        for d in (list(ligand_stats1.bonds) + list(ligand_stats1.pdb))
        for atom2 in atoms_in_ligand2
        if d.ligand.equals(atom2)
    ]

    if len(shared_atoms) != 2:
        raise ValueError(f"Expected exactly 2 shared ligand atoms, got {len(shared_atoms)}") # for now all rings are with len=4

    shared_atom1_name, shared_atom1_element = shared_atoms[0].name, shared_atoms[0].element # hardcoded due to the check above
    shared_atom2_name, shared_atom2_element = shared_atoms[1].name, shared_atoms[1].element # hardcoded due to the check above

    ring_elements = [metal1_element, shared_atom1_element, metal2_element, shared_atom2_element]
    metal_ring_count = sum(1 for element in ring_elements if element.upper() in METALS)
    fe_ring_count = sum(1 for element in ring_elements if element.upper() == "FE")
    s_ring_count = sum(1 for element in ring_elements if element.upper() == "S")

    ring_type = None
    if s_ring_count >= 2: # we assume that rings are having 2 atoms of S   
        if fe_ring_count == 1:
            ring_type = "s-fe-s-m" if metal_ring_count == 2 else "s-fe-s-nm"
        elif fe_ring_count == 2 and s_ring_count == 2:
            ring_type = "s-fe-s-fe"

    ring_stats = RingStats(
        ring_type=ring_type,
        coordination=ligand_stats1.coordination,
        other_coordination=ligand_stats2.coordination,
        class_name=ligand_stats1.clazz,
        other_class_name=ligand_stats2.clazz,
        metal_name=metal1_name,
        other_metal_name=metal2_name
    )

    return CLASSIFIER.get_angles_stats(ring_stats)

#################################################################################################################

def count_atoms(atoms: Dict[str, Any]) -> Tuple[int, int]:
    """
    Count the total number of atoms and the number of non-hydrogen atoms.

    This function expects a dictionary containing atomic data with at least two keys:
    - ATOM_ID: A key whose value is a list representing all atoms.
    - TYPE_SYMBOL: A key whose value is a list of element symbols corresponding to the atoms.

    Parameters:
        atoms (Dict[str, Any]): A dictionary containing atomic information. The dictionary should
            include the keys ATOM_ID and TYPE_SYMBOL. The ATOM_ID key's value is used to determine
            the total number of atoms, while the TYPE_SYMBOL key's value is examined to count how many
            atoms are not hydrogen (i.e., have a symbol other than "H").

    Returns:
        Tuple[int, int]: A tuple with two integers:
            - The first integer is the total number of atoms as determined by the length of atoms[ATOM_ID].
            - The second integer is the number of non-hydrogen atoms, computed by counting the entries
              in atoms[TYPE_SYMBOL] that are not equal to "H".

    Raises:
        KeyError: If either ATOM_ID or TYPE_SYMBOL is not present in the provided dictionary.
    """
    n_atoms = len(atoms[ATOM_ID])
    n_nhatoms = len([x for x in atoms[TYPE_SYMBOL] if x != "H"])
    return n_atoms, n_nhatoms

def get_category_with_warning(block: gemmi.cif.Block, 
                              category: str, 
                              category_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the MMCIF category data from a given block and logs a warning if the category is not found.

    Parameters:
        block (gemmi.cif.Block): The CIF block containing MMCIF categories.
        category (str): The key for the desired MMCIF category.
        category_name (str): A descriptive name for the category used in the warning message.

    Returns:
        Optional[Dict[str, Any]]: The MMCIF category data if found; otherwise, None.
    """
    data = block.get_mmcif_category(category)
    if not data:
        Logger().warning(
            f"mmcif category {category_name} not found. Please check the CIF file."
        )
    return data

def choose_best_pdb(name: str) -> str:
    """
    Choose and return the best PDB file name from the ligand-PDB database based on specific criteria.

    Parameters:
        name (str): An identifier key used to look up a list of candidate PDB entries in the global 'mons' dictionary.
                    Each candidate is expected to be a tuple where:
                    - The first element is the PDB file name (str).
                    - The second element represents the resolution (numeric or None).
                    - The third element is a boolean indicating occupancy.

    Returns:
        str: The file name of the selected PDB, chosen according to the following logic:
             - Candidates are sorted by favoring those with a valid occupancy flag and lower resolution (if provided,
               where missing or falsy resolution is treated as a high value, defaulting to 10000 in sorting).
             - If there are any candidates with valid occupancy, the one at the top of this filtered list is chosen.
             - Otherwise, the top candidate from the overall sorted list is used.

    Raises:
        ValueError: If the given 'name' is not present in the global 'mons' dictionary, indicating that there is no corresponding 
                    PDB file in the Ligand-PDB database.

    Notes:
        - The function logs various informational and warning messages using a Logger instance to track its execution
          and potential issues regarding resolution and occupancy.
    """
    if name not in mons:
        raise ValueError(
            (
                "There is no PDB in our Ligand-PDB database. "
                "Please specify the PDB file"
            )
        )
    Logger().info("Choosing best PDB file")
    all_candidates = sorted(
        mons[name],
        key=lambda x: (not x[2], x[1] if x[1] else 10000)
    )

    candidates = [mon for mon in all_candidates if mon[2]]

    if len(candidates) == 0:
        mon = all_candidates[0]
    else:
        mon = candidates[0]

    if mon[1] and mon[1] > 2:
        if len(candidates) == 0:
            Logger().warning(
                "There is no PDB with necessary resolution "
                "and occupancy in our Ligand-PDB database. "
                "Please specify the PDB file"
            )
        else:
            Logger().warning(
                "There is no PDB with necessary resolution "
                "in our Ligand-PDB database. "
                "Please specify the PDB file"
            )
    else:
        if len(candidates) == 0:
            Logger().warning(
                "There is no PDB with necessary occupancy "
                "in our Ligand-PDB database. "
                "Please specify the PDB file"
            )

    pdb = mon[0]
    Logger().info(f"Best PDB file is {pdb}")
    return pdb

def update_bonds(bonds: Dict[str, Any], atoms: Dict[str, Any], pdb_stats: PdbStats, block: gemmi.cif.Block) -> None:
    """
    Update bond distance information in the bonds dictionary and in the provided CIF block.

    This function iterates over each bond pair (as identified by the ATOM_ID_1 and ATOM_ID_2 entries in the bonds dictionary)
    and sets up distance-related keys (VALUE_DIST, VALUE_DIST_ESD, VALUE_DIST_NUCLEUS, VALUE_DIST_NUCLEUS_ESD) with initial
    None values if they are not already present. For each bond, the elemental type of both atoms is determined using the atoms
    dictionary and gemmi.Element. If neither atom is a metal, the bond is skipped. If both atoms are metals, the function checks
    for available distance data via MetalMetalStats and, if available, updates the bond data with the computed distance and its
    estimated standard deviation. In the case of a metal–ligand bond, the function ensures that the metal atom is correctly
    identified and retrieves the bond statistics using the pdb_stats object. Finally, the updated bonds information is set in the CIF
    block under the designated bond category.

        bonds (Dict[str, Any]): Dictionary containing bond information, including atom IDs and placeholders for distance values.
        atoms (Dict[str, Any]): Dictionary containing atomic information used to determine element names.
        pdb_stats (PdbStats): An object that provides methods to retrieve ligand bond statistics.
        block (gemmi.cif.Block): The CIF block that will be updated with the computed bond distance data.

        None: The function directly modifies the bonds dictionary and updates the CIF block.
    """
    if VALUE_DIST not in bonds:
        bonds[VALUE_DIST] = [None for _ in range(len(bonds[ATOM_ID_1]))]
        bonds[VALUE_DIST_ESD] = [None for _ in range(len(bonds[ATOM_ID_1]))]
        bonds[VALUE_DIST_NUCLEUS] = [None for _ in range(len(bonds[ATOM_ID_1]))]
        bonds[VALUE_DIST_NUCLEUS_ESD] = [None for _ in range(len(bonds[ATOM_ID_1]))]

    metal_groups_by_locant = defaultdict(list)
    for metal_stats in pdb_stats.metals:
        metal_groups_by_locant[metal_stats.locant].append(metal_stats)
    metal_groups_by_locant = dict(metal_groups_by_locant)
    

    for i, _atoms in enumerate(zip(bonds[ATOM_ID_1], bonds[ATOM_ID_2])):
        metal_name, ligand_name = _atoms

        is_metal1 = gemmi.Element(get_element_name(atoms, metal_name)).is_metal
        is_metal2 = gemmi.Element(get_element_name(atoms, ligand_name)).is_metal
        

        if not is_metal1 and not is_metal2:
            continue

        if is_metal1 and is_metal2:
            if MetalPairStatsService().has_distance_data(
                get_element_name(atoms, metal_name),
                get_element_name(atoms, ligand_name),
            ):
                distance, std = MetalPairStatsService().get_distance_between_metals(
                    get_element_name(atoms, metal_name),
                    get_element_name(atoms, ligand_name),
                )
                distance = round(distance, 3)
                std = round(std, 3)
                bonds[VALUE_DIST][i] = bonds[VALUE_DIST_NUCLEUS][i] = str(distance)
                bonds[VALUE_DIST_ESD][i] = bonds[VALUE_DIST_NUCLEUS_ESD][i] = str(std)
                    
                        
                        
                continue

        if is_metal2:
            metal_name, ligand_name = ligand_name, metal_name

        bond_stat = pdb_stats.get_ligand_distance(metal_name, ligand_name)
        if bond_stat:
            bonds[VALUE_DIST][i] = bonds[VALUE_DIST_NUCLEUS][i] = str(round(bond_stat.distance[0], 3))
            bonds[VALUE_DIST_ESD][i] = bonds[VALUE_DIST_NUCLEUS_ESD][i] = str(round(bond_stat.std[0], 3))

    block.set_mmcif_category(BOND_CATEGORY, bonds)

def init_angles(angles: Dict[str, Any], pdb_stats: PdbStats, bonds: Dict[str, Any], name: str) -> None:
    """
    Initializes angle entries in the angles dictionary using ligand angles from pdb_stats.

    For each metal in pdb_stats, if bonds exist between the metal and both ligands of an angle (verified by bond_exist),
    the function appends the component id, ligand names, measured angle, and standard deviation to the respective lists.

    Parameters:
        angles (Dict[str, Any]): Expected keys are COMP_ID, ATOM_ID_1, ATOM_ID_2, ATOM_ID_3, VALUE_ANGLE, and VALUE_ANGLE_ESD.
        pdb_stats (PdbStats): Provides metal_names() and get_ligand_angles(metal_name) methods.
        bonds (Dict[str, Any]): Bond information dictionary used to verify bonds.
        name (str): Component identifier for the angle entries.
    """
    angles[COMP_ID] = list()
    angles[ATOM_ID_1] = list()
    angles[ATOM_ID_2] = list()
    angles[ATOM_ID_3] = list()
    angles[VALUE_ANGLE] = list()
    angles[VALUE_ANGLE_ESD] = list()

    for metal_name in pdb_stats.metal_names():
        for angle in pdb_stats.get_ligand_angles(metal_name):
            if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                continue
            angles[COMP_ID].append(name)
            angles[ATOM_ID_1].append(angle.ligand1.name)
            angles[ATOM_ID_2].append(metal_name)
            angles[ATOM_ID_3].append(angle.ligand2.name)
            angles[VALUE_ANGLE].append(str(round(angle.angle, 3)))
            angles[VALUE_ANGLE_ESD].append(str(round(angle.std, 3)))


def update_angles_category(angles:Dict[str, Any], atoms: Dict[str, Any], bonds :Dict[str, Any], pdb_stats: PdbStats, name: str) -> None:
    """
    Update the angles dictionary with ligand angle statistics based on atom and bond data.

    This function processes triplets of atoms representing angles and updates their corresponding
    angle values and standard deviations if valid statistics are available from the provided pdb_stats
    object. It first checks each set of atoms to confirm that the central atom is a metal element,
    then retrieves angle data for the associated ligand atoms. If valid angle statistics are found,
    the function updates the angles dictionary accordingly. Additionally, it iterates over all metal
    atoms in pdb_stats to add any missing ligand angle entries, ensuring that the corresponding bonds
    exist as per the bonds dictionary.

    Parameters:
        angles (Dict[str, Any]): A dictionary containing lists of angle-related data. Expected keys
            include ATOM_ID_1, ATOM_ID_2, ATOM_ID_3 for atom identifiers, and VALUE_ANGLE, VALUE_ANGLE_ESD
            for angle values and their standard deviations, respectively.
        atoms (Dict[str, Any]): A dictionary mapping atom identifiers to their corresponding properties.
        bonds (Dict[str, Any]): A dictionary representing bond connectivity information between atoms.
        pdb_stats(PdbStats): An object that provides statistical data about ligand angles, including methods such as 
            get_ligand_angle(), get_ligand_angles(), and metal_names(), which supply angle measurements 
            and standard deviations.
        name (str): A string identifier used to annotate new angle entries in the angles dictionary (e.g., 
            as a component identifier).

    Returns:
        None: This function modifies the angles dictionary in place.
    """
    update_angles = []
    for i, _atoms in enumerate(
        zip(angles[ATOM_ID_1], angles[ATOM_ID_2], angles[ATOM_ID_3])
    ):
        ligand1_name, metal_name, ligand2_name = _atoms

        if not gemmi.Element(get_element_name(atoms, metal_name)).is_metal:
            continue

        angle_stat = pdb_stats.get_ligand_angle(metal_name, ligand1_name, ligand2_name)
        if angle_stat:
            angles[VALUE_ANGLE][i] = str(round(angle_stat.angle, 3))
            angles[VALUE_ANGLE_ESD][i] = str(round(angle_stat.std, 3))
            update_angles.append(code(ligand1_name, metal_name, ligand2_name))

    for metal_name in pdb_stats.metal_names():
        for angle in pdb_stats.get_ligand_angles(metal_name):
            if code(angle.ligand1.name, metal_name, angle.ligand2.name) in update_angles:
                continue

            if not bond_exist(bonds, angle.ligand1.name, metal_name) or not bond_exist(bonds, angle.ligand2.name, metal_name):
                continue

            angles[COMP_ID].append(name)
            angles[ATOM_ID_1].append(angle.ligand1.name)
            angles[ATOM_ID_2].append(metal_name)
            angles[ATOM_ID_3].append(angle.ligand2.name)
            angles[VALUE_ANGLE].append(str(round(angle.angle, 3)))
            angles[VALUE_ANGLE_ESD].append(str(round(angle.std, 3)))

def update_tetragons(name: str, angles: Dict[str, Any], monomer, v: list, 
                     pdb_stats: PdbStats) -> None:
        """
        Update angle information for tetragonal coordination cycles within a molecular structure.

        This function examines cycles (minimal rings) identified in a given vertex list ('v') and processes those
        that form a tetragon (four-membered cycle). Depending on whether the first element in the cycle is a metal
        (by checking gemmi.Element), the function assigns metal and ligand vertices accordingly. It then computes
        the angles between the metal centers and the ligand atoms using the 'get_angle' method provided on the
        'monomer' object. If both required angles are present, an adjusted angle value is calculated and either
        updated in the provided 'angles' dictionary or appended as a new entry.

        Parameters:
            name (str): An identifier for the current coordination or computed geometry entry.
            angles (Dict[str, Any]): A dictionary containing lists of atomic identifiers, angle values,
                and standard deviations. Expected keys include COMP_ID, ATOM_ID_1, ATOM_ID_2, ATOM_ID_3,
                VALUE_ANGLE, and VALUE_ANGLE_ESD.
            monomer: An object representing a molecular fragment with the following properties:
                - 'code': A string attribute used to match ligand identifiers.
                - 'get_angle': A method that accepts metal and ligand identifiers to calculate the angular value
                  between them.
            v (list): A list of vertices (or nodes) representing atoms or groups within the molecular structure.
                This list is used to identify minimal cycles in the structure.

        Returns:
            None

        Notes:
            - The function uses gemmi.Element to determine if a particular vertex represents a metal element.
            - If an angle cannot be computed or is missing, a warning is logged using the Logger() mechanism.
            - The standard deviation for the angle value is hardcoded to 5.0.
        """
        for cycle in find_minimal_cycles(v):
            if len(cycle) == 4:
                if gemmi.Element(cycle[0][1]).is_metal:
                    metal1 = cycle[0]
                    metal2 = cycle[2]
                    ligand1 = cycle[1]
                    ligand2 = cycle[3]
                else:
                    metal1 = cycle[1]
                    metal2 = cycle[3]
                    ligand1 = cycle[0]
                    ligand2 = cycle[2]
                metals_stats = [metal_stat_pdb for metal in [metal1, metal2] for metal_stat_pdb in pdb_stats.metals if metal[0] == metal_stat_pdb.metal]
                has_stats = False
                if len(metals_stats) == 2:
                    metal1_stats = metals_stats[0]
                    metal2_stats = metals_stats[1]
                    
                    ligand_stats1 = pdb_stats.get_best_class(metal1_stats.code[0])
                    ligand_stats2 = pdb_stats.get_best_class(metal2_stats.code[0])
                    
                    stat_angles = get_ring_angles_stats(metal1_stats, ligand_stats1, metal2_stats, ligand_stats2)
                    
                    if stat_angles and 'S' in stat_angles and 'angle' in stat_angles['S'] and 'std' in stat_angles['S']:
                        # angle1_stat = stat_angles[metal1[0]] # might be used further
                        # angle2_stat = stat_angles[metal2[0]] # might be used further
                        val = stat_angles['S']['angle']
                        std = stat_angles['S']['std']
                        has_stats = True
                            
                if not(has_stats):
                    angle1 = monomer.get_angle(metal1[0], ligand1, ligand2)
                    angle2 = monomer.get_angle(metal2[0], ligand1, ligand2)
    
                    if not angle1 or not angle2:
                        if not angle1:
                            Logger().warning(
                                f"Angle {ligand1[0]}- {metal1[0]}-{ligand2[0]} not found in {monomer.code}"
                            )
                        if not angle2:
                            Logger().warning(
                                f"Angle {ligand1[0]}- {metal2[0]}-{ligand2[0]} not found in {monomer.code}"
                            )
                        continue
    
                    val = (360 - angle1.angle - angle2.angle) / 2
                    std = 5.0
                    
                for ligand in [ligand1, ligand2]:
                    if monomer.code == ligand[2:]:
                        found = False
                        for i, _atoms in enumerate(
                            zip(angles[ATOM_ID_1], angles[ATOM_ID_2], angles[ATOM_ID_3])
                        ):
                            metal1_name, ligand_name, metal2_name = _atoms
                            if (
                                metal1_name == metal1[0]
                                and ligand_name == ligand[0]
                                and metal2_name == metal2[0]
                            ) or (
                                metal1_name == metal2[0]
                                and ligand_name == ligand[0]
                                and metal2_name == metal1[0]
                            ):
                                angles[VALUE_ANGLE][i] = str(round(val, 3))
                                angles[VALUE_ANGLE_ESD][i] = str(round(std, 3))
                                found = True
                                break
                        if not found:
                            angles[COMP_ID].append(name)
                            angles[ATOM_ID_1].append(metal1[0])
                            angles[ATOM_ID_2].append(ligand[0])
                            angles[ATOM_ID_3].append(metal2[0])
                            angles[VALUE_ANGLE].append(str(round(val, 3)))
                            angles[VALUE_ANGLE_ESD].append(str(round(std, 3)))
            
def update_cif(output_path, path, pdb, use_cif=False, clazz=None):
    """
    Update the CIF file with ligand information.

    Args:
    - output_path (str): The path where the updated CIF file will be saved.
    - path (str): The path to the original CIF file.
    - pdb (str): The path to the PDB file.
    - use_cif (bool): Whether to use the CIF file for classification.
    - clazz (str): Predefined class.

    Returns:
    - None
    """
    Logger().info(f"Start processing {path}")
    folder, name = os.path.split(path)
    folder = os.path.split(folder)[1]
    doc = gemmi.cif.read_file(path)

    name = extract_name_from_doc(doc)

    block = extract_block_from_doc(name, doc)

    block.name = f"comp_{name}"

    # inside update_cif function replace the $SELECTION_PLACEHOLDER$ section with:
    update_ace_drg(block, name)


    # Modified $SELECTION_PLACEHOLDER$ code:
    atoms = get_atoms_category(block)
    atoms = update_energy_in_atoms(block, atoms)
    n_atoms, n_nhatoms = count_atoms(atoms)

    bonds = get_category_with_warning(block, BOND_CATEGORY, BOND_CATEGORY)
    angles = block.get_mmcif_category(ANGLE_CATEGORY)
    
    if not contains_metal(atoms):
        Logger().info(f"No metal found in {name}")
        return
    
    if use_cif:
        pdb_stats, metal_pair_stats = find_classes_cif(name, atoms, bonds, clazz=clazz)
    else:
        if pdb is None:
            pdb = choose_best_pdb(name)

        pdb_stats, metal_pair_stats = find_classes_pdb(
            name, pdb, get_bonds(atoms, bonds), get_metal_metal_bonds(atoms, bonds), only_best=True, clazz=clazz
        )

    if not use_cif and pdb_stats.is_empty():
        raise ValueError(
            f"No coordination found for {name}  in {pdb}. Please check the PDB file"
        )

    Logger().info("Ligand updating started")
    metal_metal_json = [stat.to_dict() for stat in metal_pair_stats]
    if bonds:
        update_bonds(bonds, atoms, pdb_stats, block)
        Logger().info("Distances updated")

    if not angles:
        Logger().info(f"Creating mmcif category {ANGLE_CATEGORY}.")
        init_angles(angles, pdb_stats, bonds, name)

    else:
        # Modified $SELECTION_PLACEHOLDER$ code in the update_cif function:
        update_angles_category(angles, atoms, bonds, pdb_stats, name)

    if not pdb_stats.is_empty():
        v = []
        monomer = list(pdb_stats.monomers())[-1]
        for metal_stat in monomer.metals:
            for bond in metal_stat.get_all_distances():
                v.append((metal_stat.code, bond.ligand.code))

        Logger().info("update cycles")
    
        update_tetragons(name, angles, monomer, v, pdb_stats)

        block.set_mmcif_category(ANGLE_CATEGORY, angles)
        Logger().info("Angles updated")

        if not doc.find_block("comp_list"):
            list_block = doc.add_new_block("comp_list", 0)
            x = "."
            list_block.set_mmcif_category(
                COMP_CATEGORY,
                {
                    ID: [name],
                    THREE_LETTER_CODE: [name],
                    NAME: [name.lower()],
                    GROUP: ["."],
                    NUMBER_ATOMS_ALL: [str(n_atoms)],
                    NUMBER_ATOMS_NH: [str(n_nhatoms)],
                    DESC_LEVEL: ["."],
                },
            )

    Logger().info("Ligand update completed")

    Path(os.path.split(output_path)[0]).mkdir(exist_ok=True, parents=True)
    doc.write_file(output_path)
    report_path = output_path + ".json"
    metal_metal_path = output_path + ".metal_metal.json"
    Logger().info(f"Update written to {output_path}")
    directory = os.path.dirname(output_path)
    Path(directory).mkdir(exist_ok=True, parents=True)
    with open(report_path, "w", encoding="utf-8") as json_file:
        json.dump(pdb_stats.json(), json_file, indent=4, separators=(",", ": "))
    
    with open(metal_metal_path, "w", encoding="utf-8") as json_file:
        json.dump(metal_metal_json, json_file, indent=4, separators=(",", ": "))

    if Config().save:
        save_cods(pdb_stats, os.path.dirname(output_path))
    Logger().info(f"Report written to {report_path}")
      


def get_stats(ligand, pdb, output, clazz=None):
    """
    Retrieves statistics for a given ligand and PDB file and writes the results to a JSON file.

    Args:
        ligand (str): The name of the ligand.
        pdb (str): The path to the PDB file.
        output (str): The path to the output JSON file.
        clazz (str): Predefined class.
    Returns:
        None
    """

    if ligand:
        return get_ligand_stats(ligand, pdb, output, clazz=clazz)

    return get_stats_for_all_ligands(pdb, output, clazz=clazz)


def get_ligand_stats(ligand, pdb, output, clazz=None):
    """
    Retrieves statistics for a given ligand and PDB file and writes the results to a JSON file.

    Args:
        ligand (str): The name of the ligand.
        pdb (str): The path to the PDB file.
        output (str): The path to the output JSON file.
        clazz (str): Predefined class.

    Returns:
        None
    """
    pdb_stats, metal_pair_stats = find_classes_pdb(ligand, pdb, clazz=clazz)
    results = pdb_stats.json()
    metal_metal_json = [stat.to_dict() for stat in metal_pair_stats]

    directory = os.path.dirname(output)
    Path(directory).mkdir(exist_ok=True, parents=True)
    metal_metal_path = output + ".metal_metal.json"
    with open(output, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, separators=(",", ": "))
   
    with open(metal_metal_path, "w", encoding="utf-8") as json_file:
        json.dump(metal_metal_json, json_file, indent=4, separators=(",", ": "))
    if Config().save:
        save_cods(pdb_stats, os.path.dirname(output))

    Logger().info(f"Report written to {output}")

def get_stats_for_all_ligands(pdb, output, clazz=None):
    """
    Retrieves statistics for all ligands in a given PDB file and writes the results to a JSON file.

    Args:
        pdb (str): The path to the PDB file.
        output (str): The path to the output JSON file.
        clazz (str): Predefined class.

    Returns:
        None
    """
    monomers_with_metals = set()
    st = read_structure(pdb)
    for model in st:
        lookup = {x.atom: x for x in model.all()}
        for cra in lookup.values():
            if cra.atom.element.is_metal:
                if cra.residue.name not in monomers_with_metals:
                    monomers_with_metals.add(cra.residue.name)
    if not monomers_with_metals:
        raise RuntimeError("No metal-containing ligands found in the input model.")
    Logger().info(f"Found {len(monomers_with_metals)} metal-containing ligands.")

    if os.path.isfile(output):
        pdb_name = os.path.splitext(os.path.basename(pdb))[0]
    else:
        pdb_name = pdb
    for ligand in monomers_with_metals:
        get_stats(ligand, pdb, os.path.join(output, f"{pdb_name}_{ligand}.json"), clazz=clazz)