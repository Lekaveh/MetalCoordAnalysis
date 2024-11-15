import numpy as np

import metalCoord
import metalCoord.analysis
import metalCoord.analysis.structures
from metalCoord.analysis.data import DB


class Ligand:
    """
    Represents a ligand atom in a molecular structure.

    Attributes:
        name (str): The name of the ligand atom.
        element (str): The element of the ligand atom.
        chain (str): The chain of the ligand atom.
        residue (str): The residue of the ligand atom.
        sequence (int): The sequence number of the ligand atom.
        insertion_code (str): The insertion code of the ligand atom.
        altloc (str): The alternate location identifier of the ligand atom.
    """

    def __init__(self, ligand: object) -> None:
        self._name: str = ligand.atom.name
        self._element: str = ligand.atom.element.name
        self._chain: str = ligand.chain.name
        self._residue: str = ligand.residue.name
        self._sequence: int = ligand.residue.seqid.num
        self._icode: str = ligand.residue.seqid.icode.strip().replace('\x00', '')
        self._altloc: str = ligand.atom.altloc.strip().replace('\x00', '')
        self._symmetry: int = ligand.symmetry
        self._pos: np.ndarray = ligand.pos

    @property
    def name(self) -> str:
        """Returns the name of the ligand atom."""
        return self._name

    @property
    def element(self) -> str:
        """Returns the element of the ligand atom."""
        return self._element

    @property
    def chain(self) -> str:
        """Returns the chain of the ligand atom."""
        return self._chain

    @property
    def residue(self) -> str:
        """Returns the residue of the ligand atom."""
        return self._residue

    @property
    def sequence(self) -> int:
        """Returns the sequence number of the ligand atom."""
        return self._sequence

    @property
    def insertion_code(self) -> str:
        """Returns the insertion code of the ligand atom."""
        if self._icode == "\u0000":
            return "."
        return self._icode if self._icode else "."

    @property
    def symmetry(self) -> int:
        """Returns the symmetry of the ligand atom."""
        return self._symmetry

    @property
    def altloc(self) -> str:
        """Returns the alternate location indicator of the ligand atom."""
        return "" if self._altloc == "\u0000" else self._altloc

    @property
    def code(self) -> tuple:
        """
        Returns a tuple containing the name, element, chain, residue, sequence, altloc, and symmetry code.
        """
        return (self.name, self.element, self.chain, self.residue, self.sequence, self.altloc, self.symmetry)

    def equals(self, other: 'Ligand') -> bool:
        """
        Compares the current object with another Ligand object.

        Args:
            other (Ligand): The other Ligand object to compare.

        Returns:
            bool: True if the Ligand objects are equal, False otherwise.
        """
        return self.code == other.code

    def to_dict(self) -> dict:
        """
        Converts the object to a dictionary.

        Returns:
            dict: A dictionary representation of the ligand object.
        """
        return {
            "name": self.name,
            "element": self.element,
            "chain": self.chain,
            "residue": self.residue,
            "sequence": self.sequence,
            "icode": self.insertion_code,
            "altloc": self.altloc,
            "symmetry": self.symmetry
        }


class DistanceStats:
    """
    Represents statistics for a distance measurement.

    Attributes:
        ligand (Ligand): The ligand.
        distance (float): The distance value.
        std (float): The standard deviation value.
        distances (np.ndarray): Optional array of distances.
        procrustes_dists (np.ndarray): Optional array of procrustes distances.
        description (str): Optional description.
    """

    def __init__(self, ligand: Ligand, distance: float, std: float, distances: np.ndarray = None,
                 procrustes_dists: np.ndarray = None, description: str = "") -> None:
        self._ligand: Ligand = ligand
        self._distance: float = np.round(distance, 2).tolist()
        self._std: float = np.round(np.where(std > 0.02, std, 0.02), 2).tolist()
        self._distances: np.ndarray = distances
        self._procrustes_dists: np.ndarray = procrustes_dists
        self._description: str = description

    @property
    def ligand(self) -> Ligand:
        """Returns the ligand."""
        return self._ligand

    @property
    def distance(self) -> float:
        """Returns the distance value."""
        return self._distance

    @property
    def std(self) -> float:
        """Returns the standard deviation."""
        return self._std

    @property
    def distances(self) -> np.ndarray:
        """Returns the array of distances."""
        return self._distances

    @property
    def procrustes_dists(self) -> np.ndarray:
        """Returns the array of procrustes distances."""
        return self._procrustes_dists

    @property
    def description(self) -> str:
        """Returns the description."""
        return self._description

    def to_dict(self) -> dict:
        """
        Converts the object to a dictionary.

        Returns:
            dict: A dictionary representation of the object.
        """
        d = {
            "ligand": self.ligand.to_dict(),
            "distance": self.distance,
            "std": self.std
        }
        if self.description:
            d["description"] = self.description
        return d


class AngleStats:
    """
    Class representing angle statistics between two ligands.

    Attributes:
        ligand1 (Ligand): The first ligand.
        ligand2 (Ligand): The second ligand.
        angle (float): The angle value.
        std (float): The standard deviation.
        is_ligand (bool): Whether the object represents a ligand.
        angles (np.ndarray): List of angles.
        procrustes_dists (np.ndarray): List of procrustes distances.
    """

    def __init__(self, ligand1: Ligand, ligand2: Ligand, angle_value: float, std: float, is_ligand: bool = True,
                 angles: np.ndarray = None, procrustes_dists: np.ndarray = None) -> None:
        self._ligand1: Ligand = ligand1
        self._ligand2: Ligand = ligand2
        self._angle: float = np.round(angle_value, 2).tolist()
        self._std: float = np.round(np.where(std > 3.0, std, 3.0), 2).tolist()
        self._is_ligand: bool = is_ligand
        self._angles: np.ndarray = angles
        self._procrustes_dists: np.ndarray = procrustes_dists

    @property
    def ligand1(self) -> Ligand:
        """Returns the first ligand."""
        return self._ligand1

    @property
    def ligand2(self) -> Ligand:
        """Returns the second ligand."""
        return self._ligand2

    @property
    def angle(self) -> float:
        """Returns the angle value."""
        return self._angle

    @property
    def std(self) -> float:
        """Returns the standard deviation."""
        return self._std

    @property
    def is_ligand(self) -> bool:
        """Returns whether the object represents a ligand."""
        return self._is_ligand

    @property
    def angles(self) -> np.ndarray:
        """Returns the array of angles."""
        return self._angles

    @property
    def procrustes_dists(self) -> np.ndarray:
        """Returns the array of procrustes distances."""
        return self._procrustes_dists

    def equals(self, code1: tuple, code2: tuple) -> bool:
        """
        Checks if the ligand codes match.

        Args:
            code1 (tuple): The first ligand code.
            code2 (tuple): The second ligand code.

        Returns:
            bool: True if the ligand codes match, False otherwise.
        """
        return (self.ligand1.code == code1 and self.ligand2.code == code2) or (self.ligand1.code == code2 and self.ligand2.code == code1)

    def to_dict(self) -> dict:
        """
        Converts the object to a dictionary representation.

        Returns:
            dict: A dictionary containing the object's attributes.
        """
        return {
            "ligand1": self.ligand1.to_dict(),
            "ligand2": self.ligand2.to_dict(),
            "angle": self.angle,
            "std": self.std
        }


class LigandStats:
    """
    Represents the statistics of a ligand.

    Attributes:
        _clazz (str): The class of the ligand.
        _procrustes (float): The procrustes value of the ligand.
        _coordination (int): The coordination number of the ligand.
        _count (int): The count of the ligand.
        _bonds (list): The list of bonds associated with the ligand.
        _pdb_bonds (list): The list of PDB bonds associated with the ligand.
        _angles (list): The list of angles associated with the ligand.
        _description (str): The description of the ligand.
    """

    def __init__(self, clazz: str, procrustes: float, coordination: int, count: int, description: str) -> None:
        """
        Initializes a LigandStats object.

        Args:
            clazz (str): The class of the ligand.
            procrustes (float): The procrustes value of the ligand.
            coordination (int): The coordination number of the ligand.
            count (int): The count of the ligand.
            description (str): The description of the ligand.
        """
        self._clazz: str = clazz
        self._procrustes: float = procrustes
        self._coordination: int = coordination
        self._count: int = count
        self._bonds: list = []
        self._pdb_bonds: list = []
        self._angles: list = []
        self._description: str = description
        self._cod_files: dict = {}

    @property
    def clazz(self) -> str:
        """
        str: The class of the ligand.
        """
        return self._clazz

    @property
    def procrustes(self) -> float:
        """
        float: The procrustes value of the ligand.
        """
        return self._procrustes
    
    
    @property
    def coordination(self) -> int:
        """
        int: The coordination number of the ligand.
        """
        return self._coordination

    @property
    def count(self) -> int:
        """
        int: The count of the ligand.
        """
        return self._count

    @property
    def description(self) -> str:
        """
        str: The description of the ligand.
        """
        return self._description

    @property
    def cod_files(self) -> dict:
        """
        dict: The names of COD files associated with the ligand.
        """
        return self._cod_files.keys()

    @property
    def cods(self) -> iter:
        """
        Generator function that yields the key-value pairs of the _cod_files dictionary.

        Yields:
            tuple: A tuple containing the key and value of each item in the _cod_files dictionary.
        """
        for key, value in self._cod_files.items():
            yield key, value

    @property
    def bond_count(self) -> int:
        """
        int: The total number of bonds associated with the ligand.
        """
        return len(self._bonds) + len(self._pdb_bonds)

    @property
    def bonds(self) -> iter:
        """
        Generator: Yields each bond associated with the ligand.
        """
        for bond in self._bonds:
            yield bond

    @property
    def pdb(self) -> iter:
        """
        Generator: Yields each PDB bond associated with the ligand.
        """
        for bond in self._pdb_bonds:
            yield bond

    @property
    def angles(self) -> iter:
        """
        Generator: Yields each angle associated with the ligand.
        """
        for ligand_angle in self._angles:
            yield ligand_angle

    @property
    def ligand_angles(self) -> iter:
        """
        Generator: Yields each angle associated with the ligand that involves only ligands.
        """
        for ligand_angle in self._angles:
            if ligand_angle.is_ligand:
                yield ligand_angle

    def weighted_procrustes(self, metal: str) -> float:
        """
        Calculates the weighted procrustes score of the ligand.

        Args:
            metal (str): The metal identifier.

        Returns:
            float: The weighted procrustes score.
        """
        frequencies = DB.get_frequency_metal_ccordination(metal, self.coordination)
        freqs = [x["frequency"] for x in frequencies.values()]
        freq = frequencies.get(self.clazz, {}).get("frequency", 1e-7)
        return self.procrustes*(1 - np.exp(freq)/np.sum(freqs))

    def get_ligand_bond(self, ligand_name: str) -> DistanceStats:
        """
        Retrieves the bond associated with the specified ligand name.

        Args:
            ligand_name (str): The name of the ligand.

        Returns:
            Bond or None: The bond associated with the specified ligand name, or None if not found.
        """
        for bond in self.bonds:
            if bond.ligand.name == ligand_name:
                return bond
        return None

    def get_ligand_angle(self, ligand1_name: str, ligand2_name: str) -> AngleStats:
        """
        Retrieves the angle associated with the specified ligand names.

        Args:
            ligand1_name (str): The name of the first ligand.
            ligand2_name (str): The name of the second ligand.

        Returns:
            Angle or None: The angle associated with the specified ligand names, or None if not found.
        """
        for ligand_angle in self.ligand_angles:
            if (ligand_angle.ligand1.name == ligand1_name and ligand_angle.ligand2.name == ligand2_name) or (ligand_angle.ligand1.name == ligand2_name and ligand_angle.ligand2.name == ligand1_name):
                return ligand_angle
        return None

    def get_angle(self, ligand1_code: str, ligand2_code: str) -> AngleStats:
        """
        Retrieves the angle associated with the specified ligand codes.

        Args:
            ligand1_code (str): The code of the first ligand.
            ligand2_code (str): The code of the second ligand.

        Returns:
            Angle or None: The angle associated with the specified ligand codes, or None if not found.
        """
        for ligand_angle in self.angles:
            if ligand_angle.equals(ligand1_code, ligand2_code):
                return ligand_angle
        return None

    def add_bond(self, distance: DistanceStats) -> None:
        """
        Adds a bond to the ligand.

        Args:
            distance: The bond distance to add.
        """
        self._bonds.append(distance)

    def add_pdb_bond(self, distance: DistanceStats) -> None:
        """
        Adds a PDB bond to the ligand.

        Args:
            distance: The PDB bond distance to add.
        """
        self._pdb_bonds.append(distance)

    def add_angle(self, new_angle: DistanceStats) -> None:
        """
        Adds an angle to the ligand.

        Args:
            new_angle: The angle to add.
        """
        self._angles.append(new_angle)

    def add_cod_file(self, cod_file: str, structure: metalCoord.analysis.structures.Ligand) -> None:
        """
        Adds a COD file to the ligand.

        Args:
            cod_file: The COD file to add.
        """
        self._cod_files[cod_file] = structure

    def to_dict(self) -> dict:
        """
        Converts the LigandStats object to a dictionary.

        Returns:
            dict: The LigandStats object represented as a dictionary.
        """
        clazz = {"class": self.clazz, "procrustes": np.round(float(self.procrustes), 3), "coordination": self.coordination,  "count": self.count, "description": self.description,
                 "base": [], "angles": [], "pdb": []}
        for b in self.bonds:
            clazz["base"].append(b.to_dict())
        for a in self.angles:
            clazz["angles"].append(a.to_dict())
        for p in self.pdb:
            clazz["pdb"].append(p.to_dict())
        return clazz


class MetalStats:
    def __init__(self, structure: metalCoord.analysis.structures.Ligand) -> None:
        """
        Initialize a MetalStats object.

        Args:
            structure (object): The structure containing metal information.
        """
        self._metal: str = structure.metal.atom.name
        self._metal_element: str = str(structure.metal.element)
        self._chain: str = structure.chain.name
        self._residue: str = structure.residue.name
        self._sequence: int = structure.residue.seqid.num
        self._mean_occ: float = structure.mean_occ()
        self._mean_b: float = structure.mean_b()
        self._icode: str = structure.residue.seqid.icode.strip().replace('\x00', '')
        self._altloc: str = structure.metal.atom.altloc.strip().replace('\x00', '')
        self._pos: np.ndarray = structure.metal.pos
        self._ligands: list = []

    @property
    def code(self) -> tuple:
        """
        Get the code of the metal.

        Returns:
            tuple: A tuple containing the metal, metal element, chain, residue, and sequence.
        """
        return (self._metal, self._metal_element, self._chain, self._residue, str(self._sequence))
    
    @property
    def metal(self) -> str:
        """
        Get the metal identifier.

        Returns:
            str: The metal identifier.
        """
        return self._metal

    @property
    def metal_element(self) -> str:
        """
        Get the metal element.

        Returns:
            str: The metal element.
        """
        return self._metal_element

    @property
    def chain(self) -> str:
        """
        Get the chain identifier.

        Returns:
            str: The chain identifier.
        """
        return self._chain

    @property
    def residue(self) -> str:
        """
        Get the residue identifier.

        Returns:
            str: The residue identifier.
        """
        return self._residue

    @property
    def sequence(self) -> int:
        """
        Get the sequence number.

        Returns:
            int: The sequence number.
        """
        return self._sequence

    @property
    def altloc(self) -> str:
        """
        Get the alternative location identifier.

        Returns:
            str: The alternative location identifier.
        """
        return self._altloc

    @property
    def insertion_code(self) -> str:
        """
        Returns the insertion code of the metal.

        Returns:
            str: The insertion code of the metal.
        """
        if self._icode == "\u0000":
            return "."
        return self._icode if self._icode else "."

    @property
    def mean_occ(self) -> float:
        """
        Get the mean occupancy.

        Returns:
            float: The mean occupancy.
        """
        return self._mean_occ

    @property
    def mean_b(self) -> float:
        """
        Get the mean B-factor.

        Returns:
            float: The mean B-factor.
        """
        return self._mean_b

    @property
    def ligands(self) -> iter:
        """
        Get an iterator over the ligands.

        Yields:
            Ligand: A ligand object.
        """
        for ligand in self._ligands:
            yield ligand

    def same_metal(self, other: 'MetalStats') -> bool:
        """
        Check if two MetalStats objects represent the same metal.

        Args:
            other (MetalStats): Another MetalStats object.

        Returns:
            bool: True if the metals are the same, False otherwise.
        """
        return (self.metal == other.metal) and (self.metal_element == other.metal_element)

    def same_monomer(self, other: 'MetalStats') -> bool:
        """
        Check if two MetalStats objects belong to the same monomer.

        Args:
            other (MetalStats): Another MetalStats object.

        Returns:
            bool: True if the monomers are the same, False otherwise.
        """
        return (self.chain == other.chain) and (self.residue == other.residue) and (self.sequence == other.sequence)

    def add_ligand(self, ligand: 'Ligand') -> None:
        """
        Add a ligand to the MetalStats object.

        Args:
            ligand (Ligand): A ligand object.
        """
        self._ligands.append(ligand)

    def is_ligand_atom(self, atom: object) -> bool:
        """
        Check if an atom is a ligand atom.

        Args:
            atom (object): An atom object.

        Returns:
            bool: True if the atom is a ligand atom, False otherwise.
        """
        return (atom.chain == self.chain) and (atom.residue == self.residue) and (atom.sequence == self.sequence)

    def get_coordination(self) -> int:
        """
        Get the maximum coordination number among the ligands.

        Returns:
            int: The maximum coordination number.
        """
        return np.max([l.coordination for l in self.ligands])

    def get_best_class(self) -> 'Ligand':
        """
        Get the best class of ligands based on the coordination number and procrustes score.

        Returns:
            Ligand: The best class of ligands.
        """
        coord = self.get_coordination()
        ligands = [l for l in self._ligands if l.coordination == coord]
        weighted_procrustes = [l.weighted_procrustes(self.metal_element) for l in ligands]
        return ligands[np.argmin(weighted_procrustes)]

    def get_all_distances(self) -> list:
        """
        Get a list of all distances from the best class of ligands.

        Returns:
            list: A list of distances.
        """
        clazz = self.get_best_class()
        return list(clazz.bonds) + list(clazz.pdb)

    def get_ligand_distances(self) -> list:
        """
        Get a list of distances from the best class of ligands.

        Returns:
            list: A list of distances.
        """
        clazz = self.get_best_class()
        return list(clazz.bonds)

    def get_all_angles(self) -> list:
        """
        Get a list of all angles from the best class of ligands.

        Returns:
            list: A list of angles.
        """
        clazz = self.get_best_class()
        return list(clazz.angles)

    def get_ligand_angles(self) -> list:
        """
        Get a list of ligand angles from the best class of ligands.

        Returns:
            list: A list of ligand angles.
        """
        clazz = self.get_best_class()
        return [angle for angle in clazz.angles if self.is_ligand_atom(angle.ligand1) and self.is_ligand_atom(angle.ligand2)]

    def get_ligand_bond(self, ligand_name: str) -> 'DistanceStats':
        """
        Get the bond associated with a ligand.

        Args:
            ligand_name (str): The name of the ligand.

        Returns:
            DistanceStats: The bond associated with the ligand.
        """
        clazz = self.get_best_class()
        return clazz.get_ligand_bond(ligand_name)

    def get_ligand_angle(self, ligand1_name: str, ligand2_name: str) -> 'AngleStats':
        """
        Get the angle between two ligands.

        Args:
            ligand1_name (str): The name of the first ligand.
            ligand2_name (str): The name of the second ligand.

        Returns:
            AngleStats: The angle between the ligands.
        """
        clazz = self.get_best_class()
        return clazz.get_ligand_angle(ligand1_name, ligand2_name)

    def get_angle(self, ligand1_code: str, ligand2_code: str) -> 'AngleStats':
        """
        Get the angle between two ligands based on their codes.

        Args:
            ligand1_code (str): The code of the first ligand.
            ligand2_code (str): The code of the second ligand.

        Returns:
            AngleStats: The angle between the ligands.
        """
        clazz = self.get_best_class()
        return clazz.get_angle(ligand1_code, ligand2_code)

    def is_empty(self) -> bool:
        """
        Check if the MetalStats object has any ligands.

        Returns:
            bool: True if the object has no ligands, False otherwise.
        """
        return len(self._ligands) == 0

    def to_dict(self) -> dict:
        """
        Convert the MetalStats object to a dictionary.

        Returns:
            dict: A dictionary representation of the MetalStats object.
        """
        metal = {
            "chain": self.chain,
            "residue": self.residue,
            "sequence": self.sequence,
            "metal": self.metal,
            "metalElement": self.metal_element,
            "icode": self.insertion_code,
            "altloc": self.altloc,
            "ligands": []
        }

        for l in sorted(self.ligands, key=lambda x: (-x.coordination, x.weighted_procrustes(self.metal_element))):
            metal["ligands"].append(l.to_dict())

        return metal


class MonomerStats():
    """
    Represents the statistics of a monomer in a metal coordination analysis.

    Attributes:
        _chain (str): The chain identifier of the monomer.
        _residue (str): The residue identifier of the monomer.
        _sequence (str): The sequence of the monomer.
        _metals (dict): A dictionary containing the metals associated with the monomer.

    Methods:
        code(): Returns a tuple containing the chain, residue, and sequence of the monomer.
        chain(): Returns the chain identifier of the monomer.
        residue(): Returns the residue identifier of the monomer.
        sequence(): Returns the sequence of the monomer.
        metals(): Returns a generator that yields the metals associated with the monomer.
        metal_names(): Returns a list of the names of the metals associated with the monomer.
        get_metal(metal_name): Returns the metal with the specified name, or None if not found.
        is_in(atom): Checks if the monomer is associated with the specified atom.
        add_metal(metal): Adds a metal to the monomer if it is associated with the monomer.
        contains(metal_name): Checks if the monomer contains a metal with the specified name.
        get_best_class(metal_name): Returns the best class of the metal with the specified name, or None if not found.
        get_ligand_bond(metal_name, ligand_name): Returns the bond length between the metal and the specified ligand, or None if not found.
        get_ligand_angle(metal_name, ligand1_name, ligand2_name): Returns the angle between the metal and the specified ligands, or None if not found.
        get_angle(metal_name, ligand1_code, ligand2_code): Returns the angle between the specified ligands in the metal coordination, or None if not found.
        len(): Returns the number of metals associated with the monomer.
        is_empty(): Checks if the monomer has no associated metals.
    """

    def __init__(self, chain: str, residue: str, sequence: str) -> None:
        self._chain: str = chain
        self._residue: str = residue
        self._sequence: str = sequence
        self._metals: dict[str, MetalStats] = dict()

    @property
    def code(self) -> tuple[str, str, str]:
        """
        Returns the code of the model.

        The code is a tuple containing the chain, residue, and sequence of the model.

        Returns:
            tuple: A tuple containing the chain, residue, and sequence of the model.
        """
        return (self._chain, self._residue, self._sequence)

    @property
    def chain(self) -> str:
        """
        Get the chain of the model.

        Returns:
            str: The chain of the model.
        """
        return self._chain

    @property
    def residue(self) -> str:
        """
        Get the residue associated with this model.

        Returns:
            The residue object.
        """
        return self._residue

    @property
    def sequence(self) -> str:
        """
        Get the sequence of the monomer.

        Returns:
            str: The sequence of the monomer.
        """
        return self._sequence

    @property
    def metals(self) -> iter:
        """
        Get a generator that yields the metals associated with the monomer.

        Yields:
            Metal: The metals associated with the monomer.
        """
        for metal in self._metals.values():
            yield metal

    def metal_names(self) -> list[str]:
        """
        Get a list of the names of the metals associated with the monomer.

        Returns:
            list: A list of the names of the metals associated with the monomer.
        """
        return list(self._metals.keys())

    def get_metal(self, metal_name: str) -> MetalStats:
        """
        Get the metal with the specified name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            Metal: The metal with the specified name, or None if not found.
        """
        return self._metals.get(metal_name, None)

    def is_in(self, atom: object) -> bool:
        """
        Check if the monomer is associated with the specified atom.

        Args:
            atom (Atom): The atom to check.

        Returns:
            bool: True if the monomer is associated with the atom, False otherwise.
        """
        return self._chain == atom.chain and self._residue == atom.residue and self._sequence == atom.sequence

    def add_metal(self, metal: MetalStats) -> None:
        """
        Add a metal to the monomer if it is associated with the monomer.

        Args:
            metal (Metal): The metal to add.
        """
        if self.is_in(metal):
            self._metals.setdefault(metal.metal, metal)

    def contains(self, metal_name: str) -> bool:
        """
        Check if the monomer contains a metal with the specified name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            bool: True if the monomer contains a metal with the specified name, False otherwise.
        """
        return metal_name in self._metals

    def get_best_class(self, metal_name: str) -> str:
        """
        Get the best class of the metal with the specified name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            str: The best class of the metal with the specified name, or None if not found.
        """
        if metal_name in self._metals:
            return self._metals[metal_name].get_best_class()
        return None

    def get_ligand_bond(self, metal_name: str, ligand_name: str) -> DistanceStats:
        """
        Get the bond length between the metal and the specified ligand.

        Args:
            metal_name (str): The name of the metal.
            ligand_name (str): The name of the ligand.

        Returns:
            float: The bond length between the metal and the specified ligand, or None if not found.
        """
        if metal_name in self._metals:
            return self._metals[metal_name].get_ligand_bond(ligand_name)
        return None

    def get_ligand_angle(self, metal_name: str, ligand1_name: str, ligand2_name: str) -> AngleStats:
        """
        Get the angle between the metal and the specified ligands.

        Args:
            metal_name (str): The name of the metal.
            ligand1_name (str): The name of the first ligand.
            ligand2_name (str): The name of the second ligand.

        Returns:
            float: The angle between the metal and the specified ligands, or None if not found.
        """
        if metal_name in self._metals:
            return self._metals[metal_name].get_ligand_angle(ligand1_name, ligand2_name)
        return None

    def get_angle(self, metal_name: str, ligand1_code: str, ligand2_code: str) -> AngleStats:
        """
        Get the angle between the specified ligands in the metal coordination.

        Args:
            metal_name (str): The name of the metal.
            ligand1_code (str): The code of the first ligand.
            ligand2_code (str): The code of the second ligand.

        Returns:
            float: The angle between the specified ligands in the metal coordination, or None if not found.
        """
        if metal_name in self._metals:
            return self._metals[metal_name].get_angle(ligand1_code, ligand2_code)
        return None

    def len(self) -> int:
        """
        Get the number of metals associated with the monomer.

        Returns:
            int: The number of metals associated with the monomer.
        """
        return len(self._metals)

    def is_empty(self) -> bool:
        """
        Check if the monomer has no associated metals.

        Returns:
            bool: True if the monomer has no associated metals, False otherwise.
        """
        return self.len() == 0


class PdbStats:
    """
    Represents the statistics of a PDB file.

    Attributes:
        _monomers (dict): A dictionary containing the monomers in the PDB file.

    Methods:
        add_metal: Adds a metal to the PDB file.
        monomers: Returns an iterator over the monomers in the PDB file.
        metal_names: Returns a list of unique metal names in the PDB file.
        metals: Returns an iterator over all the metals in the PDB file.
        get_best_class: Returns the best class for a given metal name.
        get_ligand_distances: Returns the ligand distances for a given metal name.
        get_ligand_distance: Returns the ligand distance for a given metal name and ligand name.
        get_ligand_angle: Returns the ligand angle for a given metal name, ligand1 name, and ligand2 name.
        get_ligand_angles: Returns the ligand angles for a given metal name.
        get_all_distances: Returns all the distances for a given metal name.
        is_empty: Checks if the PDB file is empty.
        len: Returns the total number of monomers in the PDB file.
        json: Returns a JSON representation of the PDB file.
    """

    def __init__(self) -> None:
        self._monomers: dict[str, MonomerStats] = dict()

    def add_metal(self, metal: MetalStats) -> None:
        """
        Adds a metal to the PDB file.

        Args:
            metal: The metal to be added.

        Returns:
            None
        """
        if not metal.is_empty():
            self._monomers.setdefault(metal.chain + metal.residue + str(metal.sequence),
                                      MonomerStats(metal.chain, metal.residue, metal.sequence)).add_metal(metal)

    def monomers(self) -> iter:
        """
        Returns an iterator over the monomers in the PDB file.

        Yields:
            MonomerStats: The monomer statistics.
        """
        for monomer in self._monomers.values():
            yield monomer

    def metal_names(self) -> list[str]:
        """
        Returns a list of unique metal names in the PDB file.

        Returns:
            list: The list of unique metal names.
        """
        return np.unique([name for monomer in self._monomers.values() for name in monomer.metal_names()]).tolist()

    @property
    def metals(self) -> iter:
        """
        Returns an iterator over all the metals in the PDB file.

        Yields:
            MetalStats: The metal statistics.
        """
        for monomer in self._monomers.values():
            for metal in monomer.metals:
                yield metal

    def get_best_class(self, metal_name: str) -> LigandStats:
        """
        Returns the best class for a given metal name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            LigandStats: The best class for the given metal name.
        """
        metals = [monomer.get_metal(
            metal_name) for monomer in self.monomers() if monomer.contains(metal_name)]
        classes = [metal.get_best_class() for metal in metals]
        coordinations = [clazz.coordination for clazz in classes]
        procrustes = [clazz.procrustes for clazz in classes]

        if not classes or not coordinations:
            return None

        if np.min(coordinations) != np.max(coordinations):
            b_values = [metal.mean_b for metal in metals]
            occ_values = [metal.mean_occ for metal in metals]
            best_occ = np.max(occ_values)
            return classes[np.argmin(np.where(occ_values == best_occ, b_values, np.inf))]

        return classes[np.argmin(procrustes)]

    def get_ligand_distances(self, metal_name: str) -> list[DistanceStats]:
        """
        Returns the ligand distances for a given metal name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            list: The ligand distances for the given metal name.
        """
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_distances()
        return []

    def get_ligand_distance(self, metal_name: str, ligand_name: str) -> DistanceStats:
        """
        Returns the ligand distance for a given metal name and ligand name.

        Args:
            metal_name (str): The name of the metal.
            ligand_name (str): The name of the ligand.

        Returns:
            DistanceStats: The ligand distance for the given metal name and ligand name.
        """
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_bond(ligand_name)
        return None

    def get_ligand_angle(self, metal_name: str, ligand1_name: str, ligand2_name: str) -> AngleStats:
        """
        Returns the ligand angle for a given metal name, ligand1 name, and ligand2 name.

        Args:
            metal_name (str): The name of the metal.
            ligand1_name (str): The name of the first ligand.
            ligand2_name (str): The name of the second ligand.

        Returns:
            AngleStats: The ligand angle for the given metal name, ligand1 name, and ligand2 name.
        """
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_angle(ligand1_name, ligand2_name)
        return None

    def get_ligand_angles(self, metal_name: str) -> list[AngleStats]:
        """
        Returns the ligand angles for a given metal name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            list: The ligand angles for the given metal name.
        """
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.ligand_angles
        return []

    def get_all_distances(self, metal_name: str) -> list[DistanceStats]:
        """
        Returns all the distances for a given metal name.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            list: All the distances for the given metal name.
        """
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_all_distances()
        return []

    def is_empty(self) -> bool:
        """
        Checks if the PDB file is empty.

        Returns:
            bool: True if the PDB file is empty, False otherwise.
        """
        return len(self._monomers) == 0

    def len(self) -> int:
        """
        Returns the total number of monomers in the PDB file.

        Returns:
            int: The total number of monomers.
        """
        return np.sum([monomer.len() for monomer in self.monomers()])

    def json(self) -> list[dict]:
        """
        Returns a JSON representation of the statistics of the PDB file.

        Returns:
            list: A list of dictionaries representing the metals in the PDB file.
        """
        return [metal.to_dict() for monomer in self.monomers() for metal in monomer.metals]
