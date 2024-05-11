import numpy as np
from abc import ABC
import gemmi
import os
from pathlib import Path
from tqdm import tqdm


class PDBReader:
    """
    A class that provides methods to read PDB file data.
    """

    @staticmethod
    def get_coord(line) -> np.ndarray:
        """
        Get the coordinates from a PDB file line.

        Args:
            line (str): A line from a PDB file.

        Returns:
            np.array: An array of coordinates.
        """
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    @staticmethod
    def get_element(line) -> str:
        """
        Get the element from a PDB file line.

        Args:
            line (str): A line from a PDB file.

        Returns:
            str: The element symbol.
        """
        return line[76:78].strip()

    @staticmethod
    def get_name(line):
        """
        Get the name from a PDB file line.

        Args:
            line (str): A line from a PDB file.

        Returns:
            str: The atom name.
        """
        return line[12:16].strip()

    @staticmethod
    def is_meta(line) -> bool:
        """
        Check if a line in a PDB file represents a metal atom.

        Args:
            line (str): A line from a PDB file.

        Returns:
            bool: True if the line represents a metal atom, False otherwise.
        """
        return (line[0:4] != 'HETA') and (line[0:4] != 'ATOM')


class Atom:
    """
    A class that represents an atom in a structure.
    """

    def __init__(self, element, name, coords):
        """
        Initialize an Atom object.

        Args:
            element (str): The element symbol.
            name (str): The atom name.
            coords (np.array): An array of coordinates.
        """
        self.__name = name
        self.__coords = coords
        self.__element = element

    @property
    def name(self) -> str:
        """
        Get the atom name.

        Returns:
            str: The atom name.
        """
        return self.__name

    @property
    def coords(self) -> np.ndarray:
        """
        Get the atom coordinates.

        Returns:
            np.array: An array of coordinates.
        """
        return self.__coords

    @property
    def element(self) -> str:
        """
        Get the element symbol.

        Returns:
            str: The element symbol.
        """
        return self.__element

    @property
    def covrad(self) -> float:
        """
        Get the covalent radius of the atom.

        Returns:
            float: The covalent radius.
        """
        return gemmi.Element(self.__element).covalent_r

    @property
    def is_metal(self) -> bool:
        """
        Check if the atom is a metal.

        Returns:
            bool: True if the atom is a metal, False otherwise.
        """
        return gemmi.Element(self.__element).is_metal

    @property
    def is_h(self) -> bool:
        """
        Check if the atom is a hydrogen atom.

        Returns:
            bool: True if the atom is a hydrogen atom, False otherwise.
        """
        return self.__element == 'H'

    @staticmethod
    def read(line) -> 'Atom':
        """
        Create an Atom object from a PDB file line.

        Args:
            line (str): A line from a PDB file.

        Returns:
            Atom: An Atom object.
        """
        return Atom(PDBReader.get_element(line), PDBReader.get_name(line), PDBReader.get_coord(line))


class Structure(ABC):
    """
    An abstract base class that represents a structure.
    """

    def __init__(self, path=None):
        """
        Initialize a Structure object.

        Args:
            path (str): The path to the structure file.
        """
        self._atoms = []
        self._coords = None
        self._path = path

    def atoms(self) -> iter:
        """
        Get an iterator over the atoms in the structure.

        Yields:
            Atom: An Atom object.
        """
        for atom in self._atoms:
            yield atom

    def atom(self, name) -> 'Atom':
        """
        Get an atom by its name.

        Args:
            name (str): The atom name.

        Returns:
            Atom: An Atom object with the specified name, or None if not found.
        """
        for atom in self._atoms:
            if atom.name() == name:
                return atom
        return None

    @property
    def coords(self):
        """
        Get the coordinates of the atoms in the structure.

        Returns:
            np.array: An array of coordinates.
        """
        if self._coords is None or len(self._coords) == len(self._atoms):
            self._coords = np.vstack([atom.coords for atom in self._atoms])
        return self._coords

    @property
    def centered_coords(self) -> np.ndarray:
        """
        Get the centered coordinates of the atoms in the structure.

        Returns:
            np.array: An array of centered coordinates.
        """
        return self.coords - self.coords[0]

    @property
    def file_name(self) -> str:
        """
        Get the file name of the structure.

        Returns:
            str: The file name.
        """
        if self._path:
            return os.path.basename(self._path)
        return None

    @property
    def path(self) -> str:
        """
        Get the path to the structure file.

        Returns:
            str: The path to the structure file.
        """
        return self._path


class MetalStructure(Structure):
    """
    A class that represents a structure containing metal atoms.
    """

    def __init__(self, path=None):
        """
        Initialize a MetalStructure object.

        Args:
            path (str): The path to the structure file.
        """
        super().__init__(path=path)
        self._nonmetals = []

    @property
    def nonmetals(self) -> iter:
        """
        Get an iterator over the non-metal atoms in the structure.

        Yields:
            Atom: An Atom object representing a non-metal atom.
        """
        for nonmetal in self._nonmetals:
            yield nonmetal

    @property
    def names(self) -> iter:
        """
        Get an iterator over the names of the atoms in the structure.

        Yields:
            str: The atom name.
        """
        for atom in self._atoms:
            yield atom.name()

    def add_atom(self, atom):
        """
        Add an atom to the structure.

        Args:
            atom (Atom): An Atom object to add to the structure.
        """
        if atom is None:
            return
        self._atoms.append(atom)
        self._nonmetals.append(atom)

    def add_atoms(self, atoms):
        """
        Add multiple atoms to the structure.

        Args:
            atoms (list): A list of Atom objects to add to the structure.
        """
        if atoms is None:
            return
        self._atoms.extend(atoms)
        self._nonmetals.extend(atoms)


class SingleMetalStructure(MetalStructure):
    """
    A class that represents a structure containing a single metal atom.
    """

    @staticmethod
    def read(path) -> 'SingleMetalStructure':
        """
        Read a SingleMetalStructure from a PDB file.

        Args:
            path (str): The path to the PDB file.

        Returns:
            SingleMetalStructure: A SingleMetalStructure object.
        """
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            count = 0
            structure = None
            for line in lines:
                if PDBReader.is_meta(line):
                    continue

                if count == 0:
                    metal = Atom.read(line)
                    structure = SingleMetalStructure(metal, path=path)
                    count = count + 1
                    continue

                ligand = Atom.read(line)
                if ligand.is_h or ligand.is_metal:
                    continue
                structure.add_atom(ligand)
                count = count + 1

        return structure

    def __init__(self, metal, atoms=None, path=None):
        """
        Initialize a SingleMetalStructure object.

        Args:
            metal (Atom): The metal atom.
            atoms (list): A list of Atom objects representing the ligands.
            path (str): The path to the structure file.
        """
        super().__init__(path=path)
        if metal is None:
            raise ValueError("No metal provided")

        self._metal = metal

        self._atoms.append(metal)
        self.add_atoms(atoms)

    @property
    def metal(self) -> 'Atom':
        """
        Get the metal atom.

        Returns:
            Atom: The metal atom.
        """
        return self._metal

    @property
    def distances(self) -> np.ndarray:
        """
        Get the distances between the metal atom and the ligand atoms.

        Returns:
            np.array: An array of distances.
        """
        return np.linalg.norm(self._metal.coords - self.coords[1:], axis=1)

    @property
    def coordination(self) -> int:
        """
        Get the coordination number of the metal atom.

        Returns:
            int: The coordination number.
        """
        return len(self._nonmetals)

    def clean_by_cov_rad(self, coeff=1.3) -> 'SingleMetalStructure':
        """
        Clean the structure by removing ligand atoms that are too far from the metal atom based on their covalent radii.

        Args:
            coeff (float): A coefficient to adjust the distance threshold.

        Returns:
            SingleMetalStructure: A new SingleMetalStructure object with the cleaned structure.
        """
        distances = self.distances
        atoms = []
        for i, atom in enumerate(self._nonmetals):
            if distances[i] <= coeff * (atom.covrad + self._metal.covrad):
                atoms.append(atom)
        return SingleMetalStructure(self._metal, atoms)


class Pattern(SingleMetalStructure):
    """
    A class that represents a pattern derived from a SingleMetalStructure.
    """

    def __init__(self, structure):
        """
        Initialize a Pattern object.

        Args:
            structure (SingleMetalStructure): A SingleMetalStructure object.
        """
        super().__init__(structure.metal, path=structure.path)
        self._class = self.file_name.replace(".pdb", "").lower()
        for s in structure.nonmetals:
            self.add_atom(s)

    @property
    def cl(self) -> str:
        """
        Get the class name of the pattern.

        Returns:
            str: The class name.
        """
        return self._class


class MultipleMetalStructure(MetalStructure):
    """
    A class that represents a structure containing multiple metal atoms.
    """

    def __init__(self, metals, atoms=None):
        """
        Initialize a MultipleMetalStructure object.

        Args:
            metals (list): A list of Atom objects representing the metal atoms.
            atoms (list): A list of Atom objects representing the non-metal atoms.
        """
        super().__init__()
        if not metals:
            raise ValueError("No metals provided")
        self._metals = []
        self._metals.extend(metals)
        self.add_atoms(atoms)

    @property
    def metals(self) -> iter:
        """
        Get an iterator over the metal atoms in the structure.

        Yields:
            Atom: An Atom object representing a metal atom.
        """
        for metal in self._metals:
            yield metal

    def metal(self, name) -> 'Atom':
        """
        Get a metal atom by its name.

        Args:
            name (str): The atom name.

        Returns:
            Atom: An Atom object with the specified name, or None if not found.
        """
        for metal in self._metals:
            if metal.name() == name:
                return metal
        return None

    def distances(self, name) -> np.ndarray:
        """
        Get the distances between a metal atom and the non-metal atoms.

        Args:
            name (str): The atom name.

        Returns:
            np.array: An array of distances.
        """
        metal = self.metal(name)
        if metal:
            return np.linalg.norm(metal.coords - self.coords[len(self._metals):], axis=1)
        return None

    def all_distances(self) -> iter:
        """
        Get an iterator over the distances between each metal atom and the non-metal atoms.

        Yields:
            np.array: An array of distances.
        """
        for metal in self._metals:
            yield np.linalg.norm(metal.coords - self.coords[len(self._metals):], axis=1)


class PatternDB:
    """
    A class that represents a database of patterns.
    """

    @staticmethod
    def read(folder):
        """
        Read a PatternDB from a folder containing PDB files.

        Args:
            folder (str): The path to the folder.

        Returns:
            PatternDB: A PatternDB object.
        """
        rootdir = Path(folder)
        db = PatternDB()
        files = [str(f) for f in rootdir.glob('**/*pdb*') if f.is_file()]
        for file in tqdm(files, desc="Patterns loading", unit="p"):
            db.add(Pattern(SingleMetalStructure.read(file)))
        return db

    def __init__(self):
        self._patterns = dict()

    def add(self, pattern) :
        """
        Add a pattern to the database.

        Args:
            pattern (Pattern): A Pattern object.
        """
        coordination_patterns = self._patterns.get(pattern.coordination, [])
        coordination_patterns.append(pattern)
        self._patterns[pattern.coordination] = coordination_patterns

    def class_names(self, coordination) -> list[str]:
        """
        Get the class names of the patterns with a specific coordination number.

        Args:
            coordination (int): The coordination number.

        Returns:
            list: A list of class names.
        """
        return [pattern.cl for pattern in self._patterns.get(coordination, [])]

    def classes(self, coordination) -> list[Pattern]:
        """
        Get the patterns with a specific coordination number.

        Args:
            coordination (int): The coordination number.

        Returns:
            list: A list of Pattern objects.
        """
        return [pattern for pattern in self._patterns.get(coordination, [])]

    def query_in_coordination(self, coordination, func, *args) -> list[Pattern]:
        """
        Query the patterns with a specific coordination number using a custom function.

        Args:
            coordination (int): The coordination number.
            func (function): A function to filter the patterns.
            *args: Additional arguments for the function.

        Returns:
            list: A list of Pattern objects that satisfy the condition.
        """
        return [pattern for pattern in self._patterns.get(coordination, []) if func(pattern.cl, *args)]

    def query_names_in_coordination(self, coordination, func, *args) -> list[str]:
        """
        Query the class names of the patterns with a specific coordination number using a custom function.

        Args:
            coordination (int): The coordination number.
            func (function): A function to filter the class names.
            *args: Additional arguments for the function.

        Returns:
            list: A list of class names that satisfy the condition.
        """
        return [pattern.cl for pattern in self.query_in_coordination(coordination, func, *args)]

    def query(self, func, *args) -> list[Pattern]:
        """
        Query all patterns in the database using a custom function.

        Args:
            func (function): A function to filter the patterns.
            *args: Additional arguments for the function.

        Returns:
            list: A list of Pattern objects that satisfy the condition.
        """
        return [pattern for pattern in self.all if func(pattern.cl, *args)]

    def query_names(self, func, *args) -> list[str]:
        """
        Query the class names of all patterns in the database using a custom function.

        Args:
            func (function): A function to filter the class names.
            *args: Additional arguments for the function.

        Returns:
            list: A list of class names that satisfy the condition.
        """
        return [pattern.cl for pattern in self.query(func, *args)]

    @property
    def all(self):
        """
        Get an iterator over all patterns in the database.

        Yields:
            Pattern: A Pattern object.
        """
        for coordination in self._patterns:
            for pattern in self._patterns[coordination]:
                yield pattern

    @property
    def all_names(self):
        """
        Get an iterator over the class names of all patterns in the database.

        Yields:
            str: The class name.
        """
        for coordination in self._patterns:
            for pattern in self._patterns[coordination]:
                yield pattern.cl

    @property
    def coordinations(self):
        """
        Get the coordination numbers present in the database.

        Returns:
            list: A sorted list of coordination numbers.
        """
        return sorted(self._patterns.keys())


def clean_sm_distance(input_path, output_path, coeff=1):
    """
    Clean a PDB file by removing ligand atoms that are too far from the metal atom based on their covalent radii.

    Args:
        input_path (str): The path to the input PDB file.
        output_path (str): The path to the output PDB file.
        coeff (float): A coefficient to adjust the distance threshold.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        count = 0

        out_lines = []
        for line in lines:
            if ((line[0:4] != 'HETA') and (line[0:4] != 'ATOM')):
                out_lines.append(line)
                continue

            if count == 0:
                out_lines.append(line)
                count = count + 1
                metal = line[76:78].strip()
                metal_coord = coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                continue

            ligand = line[76:78].strip()

            if ligand == 'H':
                continue

            if gemmi.Element(ligand).is_metal:
                continue

            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

            if np.sqrt(np.sum((coord - metal_coord)**2)) > coeff*(gemmi.Element(ligand).covalent_r + gemmi.Element(metal).covalent_r):
                continue

            out_lines.append(line)
            count = count + 1

        if count < 3:
            file.close()
            return None

        with open(output_path, 'w', encoding='utf-8') as out_file:
            for line in out_lines:
                out_file.write(line)

