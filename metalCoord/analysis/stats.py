from abc import ABC, abstractmethod
import gemmi
import numpy as np
import pandas as pd
from tqdm import tqdm
import metalCoord
import metalCoord.analysis
import metalCoord.analysis.structures
from metalCoord.analysis.classes import idealClasses
from metalCoord.analysis.cluster import modes
from metalCoord.analysis.data import DB
from metalCoord.analysis.directional import calculate_stats
from metalCoord.analysis.models import AngleStats, DistanceStats, LigandStats, Ligand
from metalCoord.config import Config
from metalCoord.correspondense.procrustes import fit
from metalCoord.logging import Logger


MAX_FILES = 2000


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
    """
    Get the groups of indices for each unique atom in atoms1 and atoms2.

    Parameters:
    atoms1 (numpy.ndarray): Array of atoms for group 1.
    atoms2 (numpy.ndarray): Array of atoms for group 2.

    Returns:
    list: A list containing two sublists. The first sublist contains the groups of indices for each unique atom in atoms1.
          The second sublist contains the groups of indices for each unique atom in atoms2.
    """
    unique_atoms = np.unique(atoms1)
    group1 = []
    group2 = []
    for atom in unique_atoms:
        group1.append(np.where(atoms1 == atom)[0].tolist())
        group2.append(np.where(atoms2 == atom)[0].tolist())

    return [group1, group2]


def euclidean(coords1, coords2):
    """
    Calculates the Euclidean distance between two sets of coordinates.

    Parameters:
    coords1 (numpy.ndarray): The first set of coordinates.
    coords2 (numpy.ndarray): The second set of coordinates.

    Returns:
    float: The Euclidean distance between the two sets of coordinates.
    """
    return np.sqrt(np.sum((coords1 - coords2)**2))


def angle(metal, ligand1, ligand2):
    """
    Calculate the angle between two vectors representing the metal-ligand bonds.

    Parameters:
    metal (array-like): The coordinates of the metal atom.
    ligand1 (array-like): The coordinates of the first ligand atom.
    ligand2 (array-like): The coordinates of the second ligand atom.

    Returns:
    float: The angle between the metal-ligand1 bond and the metal-ligand2 bond in degrees.
    """
    a = metal - ligand1
    b = metal - ligand2
    a = np.array(a)/np.linalg.norm(a)
    b = np.array(b)/np.linalg.norm(b)
    cosine_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine_angle))




from abc import ABC, abstractmethod

class StatsFinder(ABC):
    """
    Abstract base class for finding statistics in metal coordination analysis.

    Attributes:
        _finder: The candidate finder object.
        _thr: The threshold value.

    """

    def __init__(self, candidateFinder) -> None:
        self._finder = candidateFinder
        self._thr = 0.3

    @abstractmethod
    def get_stats(self, structure, data, class_result):
        """
        Abstract method to get statistics.

        Args:
            structure: The structure object.
            data: The data object.
            class_result: The class result object.

        Returns:
            None

        """
        pass

    def get_ideal_angles(self, structure, class_result):
        """
        Get ideal angles for a given structure and class result.

        Args:
            structure: The structure object.
            class_result: The class result object.

        Yields:
            AngleStats: The ideal angle statistics.

        """
        n1 = structure.ligands_len
        ideal_ligand_coord = class_result.coord[class_result.index]
        ligands = list(structure.all_ligands)
        for i in range(1, structure.coordination()):
            for j in range(i + 1, structure.coordination() + 1):
                a = angle(
                    ideal_ligand_coord[0], ideal_ligand_coord[i], ideal_ligand_coord[j])
                std = 5.000
                yield AngleStats(Ligand(ligands[i - 1]), Ligand(ligands[j - 1]), a, std, is_ligand=i <= n1 and j <= n1)

    def add_ideal_angels(self, structure, class_result, clazz_stats):
        """
        Add ideal angles to the class statistics.

        Args:
            structure: The structure object.
            class_result: The class result object.
            clazz_stats: The class statistics object.

        Returns:
            ClassStats: The updated class statistics.

        """
        if class_result and idealClasses.contains(class_result.clazz):
            for ideal_angle in self.get_ideal_angles(structure, class_result):
                clazz_stats.add_angle(ideal_angle)
        return clazz_stats

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

def create_gemmi_structure(df, coords):
    """
    Create a Gemmi structure from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data for creating the structure.

    Returns:
        gemmi.Structure: The Gemmi structure object.

    """
    residue = gemmi.Residue()
    residue.name = "RES"
    residue.seqid = gemmi.SeqId(str(1))
    atom_number = 1
    coords = coords - coords[0]
    for _, row in df.iterrows():
        # Create metal residue and add metal atom
        if atom_number == 1:
            metal_atom = gemmi.Atom()
            metal_atom.name = row['MetalName']
            metal_atom.element = gemmi.Element(row['Metal'])
            # metal_atom.pos = gemmi.Position(row['MetalX'], row['MetalY'], row['MetalZ'])
            metal_atom.pos = gemmi.Position(coords[atom_number - 1, 0], coords[atom_number - 1, 2], coords[atom_number - 1, 1])
            metal_atom.serial = atom_number
            residue.add_atom(metal_atom)
            atom_number += 1
       
        # Create ligand residue and add ligand atom
        ligand_atom = gemmi.Atom()
        ligand_atom.name = row['LigandName']
        ligand_atom.element = gemmi.Element(row['Ligand'])
        # ligand_atom.pos = gemmi.Position(row['LigandX'], row['LigandY'], row['LigandZ'])
        ligand_atom.pos = gemmi.Position(coords[atom_number - 1, 0], coords[atom_number - 1, 2], coords[atom_number - 1, 1])
        ligand_atom.serial = atom_number
        residue.add_atom(ligand_atom)
        atom_number += 1

    chain = gemmi.Chain('A')
    chain.add_residue(residue)
    model = gemmi.Model('1')
    model.add_chain(chain)

    # Save to PDB
    structure = gemmi.Structure()
    structure.add_model(model)
    return structure


class StrictCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure, class_result):
        o_ligand_atoms = np.array([structure.metal.name] + structure.atoms())

        if class_result.clazz in self._classes:
            files = self._files[class_result.clazz]
            
            if Config().use_pdb:
                pattern_ligand_coord = structure.get_coord()
            else:
                pattern_ligand_coord = class_result.coord[class_result.index]

            distances = []
            procrustes_dists = []
            sum_coords = np.zeros(pattern_ligand_coord.shape)
            n = 0
            angles = []
            if len(files) > MAX_FILES:
                files = np.random.choice(files, MAX_FILES, replace=False)

            cods = {}
            
            for file in tqdm(files, desc=f"{class_result.clazz} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                m_ligand_atoms = np.insert(
                    file_data[["Ligand"]].values.ravel(), 0, structure.metal.name)

                groups = get_groups(o_ligand_atoms,  m_ligand_atoms)

                proc_dists, indices, _, rotateds = fit(
                    pattern_ligand_coord, m_ligand_coord, groups=groups, all=True)

                m = n
                for proc_dist, index, rotated in zip(proc_dists, indices, rotateds):
                    if proc_dist >= Config().procrustes_thr():
                        continue

                    sum_coords += (rotated[index] - rotated[index][0])
                    n = n + 1

                    procrustes_dists.append(proc_dist)
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis=1))[1:].tolist())
                    angles.append([angle(m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j]) for i in range(
                        1, len(pattern_ligand_coord) - 1) for j in range(i + 1, len(pattern_ligand_coord))])
                   
                if m < n:
                    cods[file] = create_gemmi_structure(file_data, rotateds[0])
            procrustes_dists = np.array(procrustes_dists)
            distances = np.array(distances).T
            angles = np.array(angles).T

            if (len(distances) > 0 and distances.shape[1] >= Config().min_sample_size):
                clazz_stats = LigandStats(
                    class_result.clazz, class_result.proc, structure.coordination(), distances.shape[1], self._finder.description())
                for file, st in cods.items():
                    clazz_stats.add_cod_file(file, st)

                sum_coords = sum_coords/n

                for i, l in enumerate(list(structure.ligands)):
                    dist, std = modes(distances[i])
                    clazz_stats.add_bond(DistanceStats(
                        Ligand(l), dist, std, distances[i], procrustes_dists))

                for i, l in enumerate(list(structure.extra_ligands)):
                    dist, std = modes(distances[i + structure.ligands_len])
                    clazz_stats.add_pdb_bond(DistanceStats(Ligand(l), dist, std, euclidean(
                        sum_coords[i + 1 + structure.ligands_len], sum_coords[0]), procrustes_dists))

                if Config().ideal_angles:
                    self.add_ideal_angels(structure, class_result, clazz_stats)
                else:
                    k = 0
                    n1 = structure.ligands_len
                    ligands = list(structure.all_ligands)
                    for i in range(structure.coordination() - 1):
                        for j in range(i + 1, structure.coordination()):
                            a, std = calculate_stats(angles[k])
                            clazz_stats.add_angle(AngleStats(Ligand(ligands[i]), Ligand(
                                ligands[j]), a, std, is_ligand=i < n1 and j < n1, angles=angles[k], procrustes_dists=procrustes_dists))
                            k += 1

                return clazz_stats
        return None


class WeekCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure, class_result):

        if class_result.clazz in self._classes:
            files = self._files[class_result.clazz]

            if Config().use_pdb:
                pattern_ligand_coord = structure.get_coord()
            else:
                pattern_ligand_coord = class_result.coord[class_result.index]

            distances = []
            lig_names = []
            if len(files) > MAX_FILES:
                files = np.random.choice(files, MAX_FILES, replace=False)

            cods = {}
            for file in tqdm(files, desc=f"{class_result.clazz} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)

                m_ligand_coord = get_coordinate(file_data)
                proc_dist, _, _, r, _ = fit(
                    pattern_ligand_coord, m_ligand_coord)

                if proc_dist < Config().procrustes_thr():
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[0] - m_ligand_coord)**2, axis=1))[1:].tolist())
                    lig_names.append(
                        file_data[["Ligand"]].values.ravel().tolist())
                    cods[file] = create_gemmi_structure(file_data, m_ligand_coord@r)

            distances = np.array(distances).T
            lig_names = np.array(lig_names).T

            if (len(distances) > 0 and distances.shape[1] >= Config().min_sample_size):

                clazz_stats = LigandStats(
                    class_result.clazz, class_result.proc, structure.coordination(), distances.shape[1], self._finder.description())
                for file, st in cods.items():
                    clazz_stats.add_cod_file(file, st)

                ligands = list(structure.ligands)

                results = {}
                for element in np.unique(lig_names):
                    element_distances = distances.ravel(
                    )[lig_names.ravel() == element]

                    if element_distances.size == 1:
                        results[element] = modes(element_distances)
                    elif element_distances.size > 1:
                        results[element] = modes(element_distances.squeeze())

                for l in ligands:
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazz_stats.add_bond(DistanceStats(Ligand(l), dist, std))
                    else:
                        clazz_stats.add_bond(
                            self._create_covalent_distance_stats(structure, l, "Covalent distance"))

                for l in structure.extra_ligands:
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazz_stats.add_pdb_bond(
                            DistanceStats(Ligand(l), dist, std))
                    else:
                        clazz_stats.add_pdb_bond(
                            self._create_covalent_distance_stats(structure, l, "Covalent distance"))

                self.add_ideal_angels(structure, class_result, clazz_stats)

                return clazz_stats
        return None


class OnlyDistanceStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        self._finder.load(structure, data)
        data = self._finder.data("")
        clazz_stats = LigandStats(class_result.clazz, class_result.proc,
                                 structure.coordination(), -1, self._finder.description())

        for l in structure.ligands:
            dist, std, count = DB.get_distance_stats(
                structure.metal.element.name, l.atom.element.name)
            if count > 0:
                clazz_stats.add_bond(DistanceStats(
                    Ligand(l), np.array([dist]), np.array([std])))
            else:
                clazz_stats.add_bond(
                    self._create_covalent_distance_stats(structure, l, "Covalent distance"))

        for l in structure.extra_ligands:
            dist, std, count = DB.get_distance_stats(
                structure.metal.element.name, l.atom.element.name)
            if count > 0:
                clazz_stats.add_pdb_bond(DistanceStats(
                    Ligand(l), np.array([dist]), np.array([std])))
            else:
                clazz_stats.add_pdb_bond(
                    self._create_covalent_distance_stats(structure, l, "Covalent distance"))

        self.add_ideal_angels(structure, class_result, clazz_stats)

        if clazz_stats.bond_count > 0:
            return clazz_stats
        return None


class CovalentStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        clazz_stats = LigandStats(class_result.clazz if class_result else "",
                                 class_result.proc if class_result else -1, structure.coordination(), -1, self._finder.description())

        for l in structure.ligands:
            clazz_stats.add_bond(
                self._create_covalent_distance_stats(structure, l))

        for l in structure.extra_ligands:
            clazz_stats.add_pdb_bond(
                self._create_covalent_distance_stats(structure, l))

        self.add_ideal_angels(structure, class_result, clazz_stats)

        return clazz_stats
