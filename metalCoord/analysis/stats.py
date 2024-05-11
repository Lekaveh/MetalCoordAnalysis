from abc import ABC, abstractmethod
import os
import gemmi
import numpy as np
import pandas as pd
from tqdm import tqdm
import metalCoord
import metalCoord.analysis
from metalCoord.analysis.classes import Classificator, idealClasses
from metalCoord.analysis.cluster import modes
from metalCoord.analysis.data import DB
from metalCoord.analysis.directional import calculate_stats
from metalCoord.analysis.structures import get_ligands
import metalCoord.analysis.structures
from metalCoord.config import Config
from metalCoord.correspondense.procrustes import fit
from metalCoord.load.rcsb import load_pdb
from metalCoord.logging import Logger
from metalCoord.analysis.utlis import elementCode, elements


MAX_FILES = 2000


def get_structures(ligand, path, bonds=None, only_best=False) -> list:
    """
    Retrieves structures contained in a specific ligand from a given file or 4-letter PDB code.

    Args:
        ligand (str): The name of the ligand to search for.
        path (str): The path to the file or 4-letter PDB code.
        bonds (dict, optional): A dictionary of bond information. Defaults to None.
        only_best (bool, optional): Flag indicating whether to return only the best structure for each metal. Defaults to False.

    Returns:
        list: A list of structures contained in the specified ligand.

    Raises:
        Exception: If an existing pdb or mmcif file path should be provided or a 4-letter pdb code.
    """
    if bonds is None:
        bonds = {}
    # Rest of the function code...

    if os.path.isfile(path):
        st = gemmi.read_structure(path)
        return get_ligands(st, ligand, bonds, only_best=only_best)

    elif len(path) == 4:
        pdb, file_type = load_pdb(path)
        if file_type == 'cif':
            cif_block = gemmi.cif.read_string(pdb)[0]
            st = gemmi.make_structure_from_block(cif_block)
        else:
            st = gemmi.read_pdb_string(pdb)
        return get_ligands(st, ligand, bonds, only_best=only_best)

    else:
        raise FileNotFoundError(
            "Existing pdb or mmcif file path should be provided or 4 letter pdb code")


def get_coordinate(file_data: pd.DataFrame) -> np.ndarray:
    """
    Get the coordinates of the metal and ligand from the given file data.

    Parameters:
    file_data (pandas.DataFrame): The data containing metal and ligand coordinates.

    Returns:
    numpy.ndarray: A 2D array containing the metal and ligand coordinates.
    """
    return np.vstack([file_data[["MetalX", "MetalY", "MetalZ"]].values[:1], file_data[["LigandX", "LigandY", "LigandZ"]].values])


def get_groups(atoms1, atoms2):
    unique_atoms = np.unique(atoms1)
    group1 = []
    group2 = []
    for atom in unique_atoms:
        group1.append(np.where(atoms1 == atom)[0].tolist())
        group2.append(np.where(atoms2 == atom)[0].tolist())

    return [group1, group2]


def euclidean(coords1, coords2):
    return np.sqrt(np.sum((coords1 - coords2)**2))


def angle(metal, ligand1, ligand2):
    a = metal - ligand1
    b = metal - ligand2
    a = np.array(a)/np.linalg.norm(a)
    b = np.array(b)/np.linalg.norm(b)
    cosine_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine_angle))


class Ligand():
    def __init__(self, ligand) -> None:
        self._name = ligand.atom.name
        self._element = ligand.atom.element.name
        self._chain = ligand.chain.name
        self._residue = ligand.residue.name
        self._sequence = ligand.residue.seqid.num
        self._altloc = ligand.atom.altloc

    @property
    def name(self):
        return self._name

    @property
    def element(self):
        return self._element

    @property
    def chain(self):
        return self._chain

    @property
    def residue(self):
        return self._residue

    @property
    def sequence(self):
        return self._sequence

    @property
    def code(self):
        return (self.name, self.element, self.chain, self.residue, self.sequence, self.altloc)

    @property
    def altloc(self):
        return "" if self._altloc == "\u0000" else self._altloc

    def equals(self, other):
        return self.code == other.code

    def to_dict(self):
        return {"name": self.name, "element": self.element, "chain": self.chain, "residue": self.residue, "sequence ": self.sequence, "altloc": self.altloc}


class DistanceStats():
    """
    Represents statistics for a distance measurement.

    Args:
        ligand (str): The ligand name.
        distance (float): The distance value.
        std (float): The standard deviation value.
        distances (list): Optional list of distances.
        procrustes_dists (list): Optional list of procrustes distances.
        description (str): Optional description.

    Attributes:
        ligand (str): The ligand name.
        distance (float): The distance value.
        std (float): The standard deviation value.
        distances (list): Optional list of distances.
        procrustes_dists (list): Optional list of procrustes distances.
        description (str): Optional description.
    """

    def __init__(self, ligand, distance, std, distances=None, procrustes_dists=None, description: str = "") -> None:
        self._ligand = ligand
        self._distance = np.round(distance, 2).tolist()
        self._std = np.round(np.where(std > 1e-02, std, 0.05), 2).tolist()
        self._distances = distances
        self._procrustes_dists = procrustes_dists
        self._description = description

    @property
    def ligand(self):
        """
        Get the ligand name.

        Returns:
            str: The ligand name.
        """
        return self._ligand

    @property
    def distance(self):
        """
        Get the distance value.

        Returns:
            float: The distance value.
        """
        return self._distance

    @property
    def std(self):
        """
        Get the standard deviation value.

        Returns:
            float: The standard deviation value.
        """
        return self._std

    @property
    def distances(self):
        """
        Get the list of distances.

        Returns:
            list: The list of distances.
        """
        return self._distances

    @property
    def procrustes_dists(self):
        """
        Get the list of procrustes distances.

        Returns:
            list: The list of procrustes distances.
        """
        return self._procrustes_dists

    @property
    def description(self):
        """
        Get the description.

        Returns:
            str: The description.
        """
        return self._description

    def to_dict(self) -> dict:
        """
        Converts the object to a dictionary representation.

        Returns:
            dict: A dictionary containing the ligand, distance, std, and description (if available).
        """
        d = {"ligand": self.ligand.to_dict(), "distance": self.distance,
             "std": self.std}
        if self.description:
            d["description"] = self.description
        return d


class AngleStats():
    """
    Class representing angle statistics between two ligands.
    """

    def __init__(self, ligand1, ligand2, angle_value, std, isLigand=True, angles=None, procrustes_dists=None) -> None:
        """
        Initialize an AngleStats object.

        Args:
            ligand1 (Ligand): The first ligand.
            ligand2 (Ligand): The second ligand.
            angle_value (float): The angle value.
            std (float): The standard deviation.
            isLigand (bool, optional): Whether the object represents a ligand. Defaults to True.
            angles (list[float], optional): List of angles. Defaults to None.
            procrustes_dists (list[float], optional): List of procrustes distances. Defaults to None.
        """
        self._ligand1 = ligand1
        self._ligand2 = ligand2
        self._angle = angle_value
        self._std = std if std > 1e-03 else 5
        self._isLigand = isLigand
        self._angles = angles
        self._procrustes_dists = procrustes_dists

    @property
    def ligand1(self):
        """
        Get the first ligand.

        Returns:
            Ligand: The first ligand.
        """
        return self._ligand1

    @property
    def ligand2(self):
        """
        Get the second ligand.

        Returns:
            Ligand: The second ligand.
        """
        return self._ligand2

    @property
    def angle(self):
        """
        Get the angle value.

        Returns:
            float: The angle value.
        """
        return self._angle

    @property
    def std(self):
        """
        Get the standard deviation.

        Returns:
            float: The standard deviation.
        """
        return self._std

    @property
    def isLigand(self):
        """
        Check if the object represents a ligand.

        Returns:
            bool: True if the object represents a ligand, False otherwise.
        """
        return self._isLigand

    @property
    def angles(self):
        """
        Get the list of angles.

        Returns:
            list[float]: List of angles.
        """
        return self._angles

    @property
    def procrustes_dists(self):
        """
        Get the list of procrustes distances.

        Returns:
            list[float]: List of procrustes distances.
        """
        return self._procrustes_dists

    def equals(self, code1, code2):
        """
        Check if the ligand codes match.

        Args:
            code1 (str): The first ligand code.
            code2 (str): The second ligand code.

        Returns:
            bool: True if the ligand codes match, False otherwise.
        """
        return (self.ligand1.code == code1 and self.ligand2.code == code2) or (self.ligand1.code == code2 and self.ligand2.code == code1)

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns:
            dict: A dictionary containing the object's attributes.
        """
        return {"ligand1": self.ligand1.to_dict(),
                "ligand2": self.ligand2.to_dict(),
                "angle": self.angle, "std": self.std}


class LigandStats():
    def __init__(self, clazz, procrustes, coordination, count, description) -> None:
        self._clazz = clazz
        self._procrustes = procrustes
        self._coordination = coordination
        self._count = count
        self._bonds = []
        self._pdb_bonds = []
        self._angles = []
        self._description = description

    @property
    def clazz(self):
        return self._clazz

    @property
    def procrustes(self):
        return self._procrustes

    @property
    def coordination(self):
        return self._coordination

    @property
    def count(self):
        return self._count

    @property
    def description(self):
        return self._description

    @property
    def bondCount(self):
        return len(self._bonds) + len(self._pdb_bonds)

    @property
    def bonds(self):
        for bond in self._bonds:
            yield bond

    @property
    def pdb(self):
        for bond in self._pdb_bonds:
            yield bond

    @property
    def angles(self):
        for angle in self._angles:
            yield angle

    @property
    def ligandAngles(self):
        for angle in self._angles:
            if angle.isLigand:
                yield angle

    def getLigandBond(self, ligand_name):
        for bond in self.bonds:
            if bond.ligand.name == ligand_name:
                return bond
        return None

    def getLigandAngle(self, ligand1_name, ligand2_name):
        for angle in self.ligandAngles:
            if (angle.ligand1.name == ligand1_name and angle.ligand2.name == ligand2_name) or (angle.ligand1.name == ligand2_name and angle.ligand2.name == ligand1_name):
                return angle
        return None

    def getAngle(self, ligand1_code, ligand2_code):
        for angle in self.angles:
            if angle.equals(ligand1_code, ligand2_code):
                return angle
        return None

    def addBond(self, distance):
        self._bonds.append(distance)

    def addPdbBond(self, distance):
        self._pdb_bonds.append(distance)

    def addAngle(self, new_angle):
        self._angles.append(new_angle)

    def to_dict(self):
        clazz = {"class": self.clazz, "procrustes": np.round(float(self.procrustes), 3), "coordination": self.coordination,  "count": self.count, "description": self.description,
                 "base": [], "angles": [], "pdb": []}
        for b in self.bonds:
            clazz["base"].append(b.to_dict())
        for a in self.angles:
            clazz["angles"].append(a.to_dict())
        for p in self.pdb:
            clazz["pdb"].append(p.to_dict())
        return clazz


class MetalStats():
    def __init__(self, metal, metalElement, chain, residue, sequence, mean_occ, mean_b) -> None:
        self._metal = metal
        self._metalElement = metalElement
        self._chain = chain
        self._residue = residue
        self._sequence = sequence
        self._mean_occ = mean_occ
        self._mean_b = mean_b
        self._ligands = []

    @property
    def code(self):
        return (self._metal, self._metalElement, self._chain, self._residue, str(self._sequence))

    @property
    def metal(self):
        return self._metal

    @property
    def metalElement(self):
        return self._metalElement

    @property
    def chain(self):
        return self._chain

    @property
    def residue(self):
        return self._residue

    @property
    def sequence(self):
        return self._sequence

    @property
    def mean_occ(self):
        return self._mean_occ

    @property
    def mean_b(self):
        return self._mean_b

    @property
    def ligands(self):
        for ligand in self._ligands:
            yield ligand

    def sameMetal(self, other):
        return (self.metal == other.metal) and (self.metalElement == other.metalElement)

    def sameMonomer(self, other):
        return (self.chain == other.chain) and (self.residue == other.residue) and (self.sequence == other.sequence)

    def addLigand(self, ligand):
        self._ligands.append(ligand)

    def isLigandAtom(self, atom):
        return (atom.chain == self.chain) and (atom.residue == self.residue) and (atom.sequence == self.sequence)

    def getCoordination(self):
        return np.max([l.coordination for l in self.ligands])

    def getBestClass(self):
        coord = self.getCoordination()
        return self._ligands[np.argmin([l.procrustes for l in self.ligands if l.coordination == coord])]

    def getAllDistances(self):
        clazz = self.getBestClass()
        return list(clazz.bonds) + list(clazz.pdb)

    def getLigandDistances(self):
        clazz = self.getBestClass()
        return list(clazz.bonds)

    def getAllAngles(self):
        clazz = self.getBestClass()
        return list(clazz.angles)

    def getLigandAngles(self):
        clazz = self.getBestClass()
        return [angle for angle in clazz.angles if self.isLigandAtom(angle.ligand1) and self.isLigandAtom(angle.ligand2)]

    def getLigandBond(self, ligand_name):
        clazz = self.getBestClass()
        return clazz.getLigandBond(ligand_name)

    def getLigandAngle(self, ligand1_name, ligand2_name):
        clazz = self.getBestClass()
        return clazz.getLigandAngle(ligand1_name, ligand2_name)

    def getAngle(self, ligand1_code, ligand2_code):
        clazz = self.getBestClass()
        return clazz.getAngle(ligand1_code, ligand2_code)

    def isEmpty(self):
        return len(self._ligands) == 0

    def to_dict(self):
        metal = {"chain": self.chain, "residue": self.residue, "sequence": self.sequence, "metal": self.metal,
                 "metalElement": self.metalElement, "ligands": []}

        for l in sorted(self.ligands, key=lambda x: (-x.coordination, x.procrustes)):
            metal["ligands"].append(l.to_dict())

        return metal


class MonomerStats():
    def __init__(self, chain, residue, sequence) -> None:
        self._chain = chain
        self._residue = residue
        self._sequence = sequence
        self._metals = dict()

    @property
    def code(self):
        return (self._chain, self._residue, self._sequence)

    @property
    def chain(self):
        return self._chain

    @property
    def residue(self):
        return self._residue

    @property
    def sequence(self):
        return self._sequence

    @property
    def metals(self):
        for metal in self._metals.values():
            yield metal

    def metalNames(self):
        return self._metals.keys()

    def getMetal(self, metal_name):
        return self._metals.get(metal_name, None)

    def isIn(self, atom):
        return self._chain == atom.chain and self._residue == atom.residue and self._sequence == atom.sequence

    def addMetal(self, metal):
        if self.isIn(metal):
            self._metals.setdefault(metal.metal, metal)

    def contains(self, metal_name):
        return metal_name in self._metals

    def getBestClass(self, metal_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getBestClass()
        return None

    def getLigandBond(self, metal_name, ligand_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getLigandBond(ligand_name)
        return None

    def getLigandAngle(self, metal_name, ligand1_name, ligand2_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getLigandAngle(ligand1_name, ligand2_name)
        return None

    def getAngle(self, metal_name, ligand1_code, ligand2_code):
        if metal_name in self._metals:
            return self._metals[metal_name].getAngle(ligand1_code, ligand2_code)
        return None

    def len(self):
        return len(self._metals)

    def isEmpty(self):
        return self.len() == 0


class PdbStats():
    def __init__(self) -> None:
        self._monomers = dict()

    def addMetal(self, metal):
        if not metal.isEmpty():
            self._monomers.setdefault(metal.chain + metal.residue + str(metal.sequence),
                                      MonomerStats(metal.chain, metal.residue, metal.sequence)).addMetal(metal)

    def monomers(self):
        for monomer in self._monomers.values():
            yield monomer

    def metalNames(self):
        return np.unique([name for monomer in self._monomers.values() for name in monomer.metalNames()]).tolist()

    @property
    def metals(self):
        for monomer in self._monomers.values():
            for metal in monomer.metals:
                yield metal

    def getBestClass(self, metal_name):

        metals = [monomer.getMetal(
            metal_name) for monomer in self.monomers() if monomer.contains(metal_name)]
        classes = [metal.getBestClass() for metal in metals]
        coordinations = [clazz.coordination for clazz in classes]
        procrustes = [clazz.procrustes for clazz in classes]

        if not classes or not coordinations:
            return None

        if np.min(coordinations) != np.max(coordinations):

            b_vlaues = [metal.mean_b for metal in metals]
            occ_values = [metal.mean_occ for metal in metals]
            best_occ = np.max(occ_values)
            return classes[np.argmin(np.where(occ_values == best_occ, b_vlaues, np.inf))]

        return classes[np.argmin(procrustes)]

    def getLigandDistances(self, metal_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandDistances()
        return []

    def getLigandDistance(self, metal_name, ligand_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandBond(ligand_name)
        return None

    def getLigandAngle(self, metal_name, ligand1_name, ligand2_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandAngle(ligand1_name, ligand2_name)
        return []

    def getLigandAngles(self, metal_name):
        clazz = self.getBestClass(metal_name)

        if clazz:
            return clazz.ligandAngles
        return []

    def getAllDistances(self, metal_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getAllDistances()
        return []

    def len(self):
        return len(self._metals)

    def isEmpty(self):
        return len(self._monomers) == 0

    def len(self):
        return np.sum([monomer.len() for monomer in self.monomers()])

    def json(self):
        return [metal.to_dict() for monomer in self.monomers() for metal in monomer.metals]


class CandidateFinder(ABC):

    def __init__(self) -> None:
        self._description = ""
        self._classes = None
        self._files = None
        self._selection = None

    def load(self, structure, data):
        self._structure = structure
        self._data = data
        self._load()
        self._classes = self._selection.Class.unique()
        self._files = {cl: self._selection[self._selection.Class ==
                                           cl].File.unique() for cl in self._classes}

    @abstractmethod
    def _load(self):
        pass

    def classes(self):
        return self._classes

    def files(self):
        return self._files

    def data(self, file):
        return self._selection[self._selection.File == file] if self._selection is not None else None

    def description(self):
        return self._description


class StrictCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Strict correspondence"

    def _load(self):
        self._selection = self._data[self._data.Code ==
                                     self._structure.code()]


class ElementCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on coordination, atom availability and atom count"

    def _load(self):
        code = elementCode(self._structure.code())
        self._selection = self._data[(self._data.ElementCode == code) & (
            self._data.Coordination == self._structure.coordination())]


class ElementInCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination and all atoms availability only"

    def _load(self):
        code = elements(self._structure.code())
        coordinationData = self._data[(
            self._data.Coordination == self._structure.coordination())]
        self._selection = coordinationData[np.all(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]


class AnyElementCandidateFinder(CandidateFinder):
    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination and at least one atom availability"

    def _load(self):
        code = elements(self._structure.code())[1:]
        coordinationData = self._data[(self._data.Coordination == self._structure.coordination()) & (
            self._data.Metal == self._structure.metal.element.name)]
        self._selection = coordinationData[np.any(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]


class NoCoordinationCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on atom availability only"

    def _load(self):
        self._selection = self._data[(
            self._data.Metal == self._structure.metal.element.name)]


class CovalentCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on covalent distances"

    def _load(self):
        self._selection = self._data[self._data.Metal ==
                                     self._structure.metal.element.name]


class StatsFinder(ABC):
    def __init__(self, candidateFinder) -> None:
        self._finder = candidateFinder
        self._thr = 0.3

    @abstractmethod
    def get_stats(self, structure, data, class_result):
        pass

    def get_ideal_angles(self, structure, class_result):

        n1 = structure.ligands_len
        ideal_ligand_coord = class_result.coord[class_result.index]
        ligands = list(structure.all_ligands)
        for i in range(1, structure.coordination()):
            for j in range(i + 1, structure.coordination() + 1):
                a = angle(
                    ideal_ligand_coord[0], ideal_ligand_coord[i], ideal_ligand_coord[j])
                std = 5.000
                yield AngleStats(Ligand(ligands[i - 1]), Ligand(ligands[j - 1]), a, std, isLigand=i <= n1 and j <= n1)

    def add_ideal_angels(self, structure, class_result, clazzStats):
        if class_result and idealClasses.contains(class_result.clazz):
            for ideal_angle in self.get_ideal_angles(structure, class_result):
                clazzStats.addAngle(ideal_angle)
        return clazzStats

    def _create_covalent_distance_stats(self, structure: metalCoord.analysis.structures.Ligand, l: metalCoord.analysis.structures.Atom, description: str = "") -> DistanceStats:
        """
        Create covalent distance statistics for a ligand and an atom.

        Args:
            structure (metalCoord.analysis.structures.Ligand): The ligand structure.
            l (metalCoord.analysis.structures.Atom): The atom.
            description (str, optional): The description. Defaults to "".

        Returns:
            DistanceStats: The covalent distance statistics.

        """
        return DistanceStats(Ligand(l), np.array([gemmi.Element(l.atom.element.name).covalent_r + gemmi.Element(structure.metal.element.name).covalent_r]), np.array([0.2]), description=description)


class FileStatsFinder(StatsFinder):
    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        self._prepare(structure, data)
        return self._calculate(structure, class_result)

    def _prepare(self, structure, data):
        self._finder.load(structure, data)
        self._classes = self._finder.classes()
        self._files = self._finder.files()

    @abstractmethod
    def _calculate(self, stucture, clazz, main_proc_dist):
        pass


class StrictCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure, class_result):
        o_ligand_atoms = np.array([structure.metal.name] + structure.atoms())

        if class_result.clazz in self._classes:
            files = self._files[class_result.clazz]

            ideal_ligand_coord = class_result.coord[class_result.index]

            distances = []
            procrustes_dists = []
            sum_coords = np.zeros(ideal_ligand_coord.shape)
            n = 0
            angles = []
            if len(files) > MAX_FILES:
                files = np.random.choice(files, MAX_FILES, replace=False)

            for file in tqdm(files, desc=f"{class_result.clazz} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                m_ligand_atoms = np.insert(
                    file_data[["Ligand"]].values.ravel(), 0, structure.metal.name)

                groups = get_groups(o_ligand_atoms,  m_ligand_atoms)

                proc_dists, indices, _, rotateds = fit(
                    ideal_ligand_coord, m_ligand_coord, groups=groups, all=True)

                for proc_dist, index, rotated in zip(proc_dists, indices, rotateds):
                    if proc_dist >= Config().procrustes_thr():
                        continue

                    sum_coords += (rotated[index] - rotated[index][0])
                    n = n + 1

                    procrustes_dists.append(proc_dist)
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis=1))[1:].tolist())
                    angles.append([angle(m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j]) for i in range(
                        1, len(ideal_ligand_coord) - 1) for j in range(i + 1, len(ideal_ligand_coord))])

            procrustes_dists = np.array(procrustes_dists)
            distances = np.array(distances).T
            angles = np.array(angles).T

            if (len(distances) > 0 and distances.shape[1] >= Config().min_sample_size):
                clazzStats = LigandStats(
                    class_result.clazz, class_result.proc, structure.coordination(), distances.shape[1], self._finder.description())

                sum_coords = sum_coords/n

                for i, l in enumerate(list(structure.ligands)):
                    dist, std = modes(distances[i])
                    clazzStats.addBond(DistanceStats(
                        Ligand(l), dist, std, distances[i], procrustes_dists))

                for i, l in enumerate(list(structure.extra_ligands)):
                    dist, std = modes(distances[i + structure.ligands_len])
                    clazzStats.addPdbBond(DistanceStats(Ligand(l), dist, std, euclidean(
                        sum_coords[i + 1 + structure.ligands_len], sum_coords[0]), procrustes_dists))

                if Config().ideal_angles:
                    self.add_ideal_angels(structure, class_result, clazzStats)
                else:
                    k = 0
                    n1 = structure.ligands_len
                    ligands = list(structure.all_ligands)
                    for i in range(structure.coordination() - 1):
                        for j in range(i + 1, structure.coordination()):
                            a, std = calculate_stats(angles[k])
                            clazzStats.addAngle(AngleStats(Ligand(ligands[i]), Ligand(
                                ligands[j]), a, std, isLigand=i < n1 and j < n1, angles=angles[k], procrustes_dists=procrustes_dists))
                            k += 1

                return clazzStats
        return None


class WeekCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure, class_result):

        if class_result.clazz in self._classes:
            files = self._files[class_result.clazz]

            ideal_ligand_coord = class_result.coord[class_result.index]

            distances = []
            ligNames = []
            if len(files) > MAX_FILES:
                files = np.random.choice(files, MAX_FILES, replace=False)

            for file in tqdm(files, desc=f"{class_result.clazz} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)

                m_ligand_coord = get_coordinate(file_data)
                proc_dist, _, _, _, _ = fit(
                    ideal_ligand_coord, m_ligand_coord)

                if proc_dist < Config().procrustes_thr():
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[0] - m_ligand_coord)**2, axis=1))[1:].tolist())
                    ligNames.append(
                        file_data[["Ligand"]].values.ravel().tolist())

            distances = np.array(distances).T
            ligNames = np.array(ligNames).T

            if (len(distances) > 0 and distances.shape[1] >= Config().min_sample_size):

                clazzStats = LigandStats(
                    class_result.clazz, class_result.proc, structure.coordination(), distances.shape[1], self._finder.description())
                ligands = list(structure.ligands)

                results = {}
                for element in np.unique(ligNames):
                    elementDistances = distances.ravel(
                    )[ligNames.ravel() == element]

                    if elementDistances.size == 1:
                        results[element] = modes(elementDistances)
                    elif elementDistances.size > 1:
                        results[element] = modes(elementDistances.squeeze())

                for i, l in enumerate(ligands):
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazzStats.addBond(DistanceStats(Ligand(l), dist, std))
                    else:
                        clazzStats.addBond(
                            self._create_covalent_distance_stats(structure, l, "Covalent distance"))

                for i, l in enumerate(structure.extra_ligands):
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazzStats.addPdbBond(
                            DistanceStats(Ligand(l), dist, std))
                    else:
                        clazzStats.addPdbBond(
                            self._create_covalent_distance_stats(structure, l, "Covalent distance"))

                self.add_ideal_angels(structure, class_result, clazzStats)

                return clazzStats
        return None


class OnlyDistanceStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        self._finder.load(structure, data)
        data = self._finder.data("")
        clazzStats = LigandStats(class_result.clazz, class_result.proc,
                                 structure.coordination(), -1, self._finder.description())

        for l in structure.ligands:
            dist, std, count = DB.getDistanceStats(
                structure.metal.element.name, l.atom.element.name)
            if count > 0:
                clazzStats.addBond(DistanceStats(
                    Ligand(l), np.array([dist]), np.array([std])))
            else:
                clazzStats.addBond(
                    self._create_covalent_distance_stats(structure, l, "Covalent distance"))

        for l in structure.extra_ligands:
            dist, std, count = DB.getDistanceStats(
                structure.metal.element.name, l.atom.element.name)
            if count > 0:
                clazzStats.addPdbBond(DistanceStats(
                    Ligand(l), np.array([dist]), np.array([std])))
            else:
                clazzStats.addPdbBond(
                    self._create_covalent_distance_stats(structure, l, "Covalent distance"))

        self.add_ideal_angels(structure, class_result, clazzStats)

        if clazzStats.bondCount > 0:
            return clazzStats
        return None


class CovalentStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        clazzStats = LigandStats(class_result.clazz if class_result else "",
                                 class_result.proc if class_result else -1, structure.coordination(), -1, self._finder.description())

        for l in structure.ligands:
            clazzStats.addBond(
                self._create_covalent_distance_stats(structure, l))

        for l in structure.extra_ligands:
            clazzStats.addPdbBond(
                self._create_covalent_distance_stats(structure, l))

        self.add_ideal_angels(structure, class_result, clazzStats)

        return clazzStats


convalent_strategy = CovalentStatsFinder(CovalentCandidateFinder())
strategies = [StrictCorrespondenceStatsFinder(StrictCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementCandidateFinder()),
              WeekCorrespondenceStatsFinder(ElementInCandidateFinder()),
              WeekCorrespondenceStatsFinder(AnyElementCandidateFinder()),
              OnlyDistanceStatsFinder(NoCoordinationCandidateFinder()),
              convalent_strategy
              ]


def find_classes(ligand, pdb_name, bonds={}, only_best=False):
    Logger().info(f"Analysing structres in  {pdb_name} for patterns")
    structures = get_structures(ligand, pdb_name, bonds, only_best)
    for structure in tqdm(structures):
        Logger().info(
            f"Structure for {structure} found. Coordination number: {structure.coordination()}")
    Logger().info(f"{len(structures)} structures found.")
    results = PdbStats()
    classificator = Classificator()

    classes = []
    for structure in tqdm(structures, desc="Structures", position=0, disable=Logger().disabled):
        structure_classes = []
        for class_result in tqdm(classificator.classify(structure), desc="Coordination", position=1, leave=False, disable=Logger().disabled):
            structure_classes.append(class_result)
        classes.append(structure_classes)

    for structure, structure_classes in zip(structures, classes):
        candidantes = []
        for class_result in structure_classes:
            candidantes.append(class_result.clazz)
        Logger().info(f"Candidates for {structure} : {candidantes}")

    for i, structure in tqdm(list(enumerate(structures)), desc="Structures", position=0, disable=Logger().disabled):
        metalStats = MetalStats(structure.metal.name, structure.metal.element.name, structure.chain.name,
                                structure.residue.name, structure.residue.seqid.num, structure.mean_occ(), structure.mean_b())
        if classes[i]:
            for class_result in classes[i]:
                for strategy in tqdm(strategies, desc="Strategies", position=1, leave=False, disable=Logger().disabled):
                    ligandStats = strategy.get_stats(
                        structure, DB.data(), class_result)
                    if ligandStats:
                        metalStats.addLigand(ligandStats)
                        break
        else:
            ligandStats = convalent_strategy.get_stats(
                structure, DB.data(), None)
            metalStats.addLigand(ligandStats)

        if not metalStats.isEmpty():
            results.addMetal(metalStats)
    Logger().info(
        f"Analysis completed. Statistics for {int(results.len())} ligands(metals) found.")
    return results
