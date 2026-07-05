from abc import ABC, abstractmethod

import gemmi
import numpy as np
import pandas as pd
from tqdm import tqdm

import metalCoord
import metalCoord.analysis
import metalCoord.analysis.structures
from metalCoord.analysis.classes import ClassificationResult, idealClasses
from metalCoord.analysis.cluster import modes
from metalCoord.analysis.data import DB
from metalCoord.analysis.directional import calculate_stats
from metalCoord.analysis.grouping import _pos_array, get_grouped_view
from metalCoord.analysis.models import (
    MIN_DISTANCE_STD,
    AngleStats,
    Atom,
    Centroid,
    DistanceStats,
    GroupedAngleStats,
    GroupedDistanceStats,
    LigandStats,
)
from metalCoord.config import Config
from metalCoord.correspondense.procrustes import fit
from metalCoord.logging import Logger


MAX_FILES = Config().max_sample_size if Config().max_sample_size else 2000


def get_coordinate(file_data: pd.DataFrame) -> np.ndarray:
    """
    Get the coordinates of the metal and ligand from the given file data.

    Parameters:
    file_data (pandas.DataFrame): The data containing metal and ligand coordinates.

    Returns:
    numpy.ndarray: A 2D array containing the metal and ligand coordinates.
    """
    return np.vstack(
        [
            file_data[["MetalX", "MetalY", "MetalZ"]].values[:1],
            file_data[["LigandX", "LigandY", "LigandZ"]].values,
        ]
    )


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
    return np.sqrt(np.sum((coords1 - coords2) ** 2))


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
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    cosine_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine_angle))


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
                    ideal_ligand_coord[0], ideal_ligand_coord[i], ideal_ligand_coord[j]
                )
                std = 5.000
                yield AngleStats(
                    Atom(ligands[i - 1]),
                    Atom(ligands[j - 1]),
                    a,
                    std,
                    is_ligand=i <= n1 and j <= n1,
                )

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

    def _create_covalent_distance_stats(
        self,
        structure: metalCoord.analysis.structures.Ligand,
        l: metalCoord.analysis.structures.Atom,
        description: str = "",
    ) -> DistanceStats:
        """
        Create covalent distance statistics for a ligand and an atom.

        Args:
            structure (metalCoord.analysis.structures.Ligand): The ligand structure.
            l (metalCoord.analysis.structures.Atom): The atom.
            description (str, optional): The description. Defaults to "".

        Returns:
            DistanceStats: The covalent distance statistics.

        """
        return DistanceStats(
            Atom(l),
            np.array(
                [
                    gemmi.Element(l.atom.element.name).covalent_r
                    + gemmi.Element(structure.metal.atom.element.name).covalent_r
                ]
            ),
            np.array([0.2]),
            description=description,
        )


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
            metal_atom.name = row["MetalName"]
            metal_atom.element = gemmi.Element(row["Metal"])
            # metal_atom.pos = gemmi.Position(row['MetalX'], row['MetalY'], row['MetalZ'])
            metal_atom.pos = gemmi.Position(
                coords[atom_number - 1, 0],
                coords[atom_number - 1, 2],
                coords[atom_number - 1, 1],
            )
            metal_atom.serial = atom_number
            residue.add_atom(metal_atom)
            atom_number += 1

        # Create ligand residue and add ligand atom
        ligand_atom = gemmi.Atom()
        ligand_atom.name = row["LigandName"]
        ligand_atom.element = gemmi.Element(row["Ligand"])
        # ligand_atom.pos = gemmi.Position(row['LigandX'], row['LigandY'], row['LigandZ'])
        ligand_atom.pos = gemmi.Position(
            coords[atom_number - 1, 0],
            coords[atom_number - 1, 2],
            coords[atom_number - 1, 1],
        )
        ligand_atom.serial = atom_number
        residue.add_atom(ligand_atom)
        atom_number += 1

    chain = gemmi.Chain("A")
    chain.add_residue(residue)
    model = gemmi.Model("1")
    model.add_chain(chain)

    # Save to PDB
    structure = gemmi.Structure()
    structure.add_model(model)
    return structure


def create_descriptor(
    class_result: ClassificationResult, structure: metalCoord.analysis.structures.Ligand
) -> str:
    """
    Create a compact descriptor string for a classified ligand structure.

    The descriptor has the form:
        "@<class_code>{atom1,atom2,...}"

    Where:
    - <class_code> is obtained from idealClasses.get_class_code(class_result.clazz).
    - The atom list is constructed from the ligand structure with the metal element placed first,
      then the ligand atoms as returned by structure.atoms().
    - The final ordering of atoms in the descriptor is determined by sorting class_result.index
      in ascending order and applying that permutation to the atom list.

    Parameters
    ----------
    class_result : ClassificationResult
        An object that must provide at least two attributes:
        - index: an iterable (e.g. list or numpy array) of integers used to determine the ordering.
        - clazz: a classification identifier passed to idealClasses.get_class_code().
    structure : metalCoord.analysis.structures.Ligand
        A ligand structure object that must provide:
        - metal.element: a string for the metal element symbol/name.
        - atoms(): a callable that returns an iterable of atom labels (strings) for the ligand.

    Returns
    -------
    str
        The formatted descriptor string, e.g. "@A{Fe,N,C,O}" (actual class code depends on idealClasses).

    Raises
    ------
    TypeError
        If the provided arguments do not have the expected attributes or types.
    ValueError
        If the length/indices in class_result.index are incompatible with the number of
        atoms produced from the structure (for example, mismatched lengths when
        reindexing).

    Notes
    -----
    - This function relies on numpy.argsort being available to compute the sort
      permutation.
    - Atoms in the returned descriptor are comma-separated with no extra spaces.
    - The function expects idealClasses to be available in the module scope and to implement
      get_class_code(clazz) -> str.

    Example
    -------
    Assuming:
        class_result.index = [2, 0, 1]
        class_result.clazz = 5
        structure.metal.element == "Fe"
        structure.atoms() -> ["C", "N", "O"]

    Then the produced descriptor will be:
        "@<code>{Fe,O,C,N}"
    where "<code>" is the result of idealClasses.get_class_code(5).
    """
    atom_names_with_symmetries = structure.atom_names_with_symmetries()
    element_names = structure.element_names()
    inv_index = class_result.lexicographic_order(
        atom_names_with_symmetries, element_names
    )
    atoms = np.array(element_names)
    class_code = idealClasses.get_class_code(class_result.clazz)
    ordered_elements = atoms[inv_index].tolist()
    descriptor = f"@{class_code}" + "{" + f"{','.join(ordered_elements)}" + "}"

    if Config().debug and Config().debug_recorder:
        icode = structure.residue.seqid.icode.strip().replace("\x00", "")
        Config().debug_recorder.add_descriptor_candidate(
            {
                "metal_site": {
                    "metal": structure.metal.atom.name,
                    "metalElement": str(structure.metal.element),
                    "chain": structure.chain.name,
                    "residue": structure.residue.name,
                    "sequence": structure.residue.seqid.num,
                    "icode": icode if icode else ".",
                    "altloc": structure.metal.atom.altloc.strip().replace("\x00", ""),
                },
                "class": class_result.clazz,
                "class_code": class_code,
                "descriptor": descriptor,
                "ordered_elements": ordered_elements,
                "index_mapping": inv_index.tolist(),
                "atom_names_with_symmetries": atom_names_with_symmetries,
                "element_names": element_names,
                "procrustes": float(class_result.proc),
            }
        )

    return descriptor


def _create_covalent_distance_stats(
    structure: metalCoord.analysis.structures.Ligand,
    ligand: metalCoord.analysis.structures.IAtom,
    description: str = "",
) -> DistanceStats:
    return DistanceStats(
        Atom(ligand),
        np.array(
            [
                gemmi.Element(ligand.atom.element.name).covalent_r
                + gemmi.Element(structure.metal.atom.element.name).covalent_r
            ]
        ),
        np.array([0.2]),
        description=description,
    )


def _create_fallback_distance_stats(
    structure: metalCoord.analysis.structures.Ligand,
    ligand: metalCoord.analysis.structures.IAtom,
) -> DistanceStats:
    dist, std, count = DB.get_distance_stats(
        structure.metal.atom.element.name, ligand.atom.element.name
    )
    if count > 0:
        return DistanceStats(Atom(ligand), np.array([dist]), np.array([std]))
    return _create_covalent_distance_stats(structure, ligand, "Covalent distance")


def _ideal_distance_raw(metal_elem: str, ligand_elem: str) -> tuple:
    """
    Raw (unrounded) ideal metal-ligand distance and its std for the generalised
    centroid geometry. Reuses the DB / covalent-radius fallback logic of
    ``_create_fallback_distance_stats`` but returns floats (not a rounded
    ``DistanceStats``); a non-finite DB std is floored to ``MIN_DISTANCE_STD``.
    """
    dist, std, count = DB.get_distance_stats(metal_elem, ligand_elem)
    if count > 0:
        std = float(std) if np.isfinite(std) and std > 0 else MIN_DISTANCE_STD
        return float(dist), std
    covalent = (
        gemmi.Element(ligand_elem).covalent_r
        + gemmi.Element(metal_elem).covalent_r
    )
    return float(covalent), 0.2


# Upper bound on the propagated centroid-distance std: as d0 -> 0 the
# sensitivity d_i/(n*d0) diverges, so cap the std at an effectively-loose value
# rather than emitting a nonsensically large (or exploding) uncertainty.
_MAX_CENTROID_STD = 1.0


def _centroid_distance(d, sigma, c):
    """
    Generalised (Apollonius) metal-to-centroid distance for a group.

    Args:
        d: ideal metal-member distances (raw), length n.
        sigma: their standard deviations (raw), length n.
        c: symmetric (n, n) matrix of observed inter-member distances.

    Returns:
        tuple (d0, sigma0, R_raw) of raw floats, or (None, None, None) when any
        input is non-finite. The radicand is clamped to 0 when negative while
        R_raw carries the raw (possibly negative) value; sigma0 is set to
        MIN_DISTANCE_STD directly when d0 == 0 (no divide-by-zero).
    """
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(d)

    upper = np.triu_indices(n, k=1)
    c_vals = c[upper]
    if not (
        np.all(np.isfinite(d))
        and np.all(np.isfinite(sigma))
        and np.all(np.isfinite(c_vals))
    ):
        return None, None, None

    r_raw = float(np.sum(d ** 2) / n - np.sum(c_vals ** 2) / (n ** 2))
    radicand = r_raw if r_raw > 0 else 0.0
    d0 = float(np.sqrt(radicand))

    if d0 <= 1e-9:
        sigma0 = MIN_DISTANCE_STD
    else:
        sigma0 = float(np.sqrt(np.sum((d / (n * d0)) ** 2 * sigma ** 2)))
        sigma0 = min(max(sigma0, MIN_DISTANCE_STD), _MAX_CENTROID_STD)
    return d0, sigma0, r_raw


def _grouped_centroid_distance(structure, members):
    """
    Metal-to-centroid distance and its uncertainty for a collapsed group.
    Returns (d0, sigma0) as raw floats, or (None, None) when the restraint must
    be skipped (non-finite inputs). Logs a warning when the ideal and observed
    geometries are materially inconsistent (radicand well below zero).
    """
    metal_elem = structure.metal.atom.element.name
    n = len(members)

    d = np.empty(n)
    sigma = np.empty(n)
    for i, member in enumerate(members):
        d[i], sigma[i] = _ideal_distance_raw(metal_elem, member.atom.element.name)

    c = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c[i, j] = c[j, i] = euclidean(
                _pos_array(members[i]), _pos_array(members[j])
            )

    d0, sigma0, r_raw = _centroid_distance(d, sigma, c)
    names = [member.atom.name for member in members]
    if d0 is None:
        Logger().warning(
            f"Grouped centroid distance skipped for metal {metal_elem} "
            f"group {names}: non-finite ideal distance or coordinate."
        )
        return None, None
    if r_raw < -1e-6:
        Logger().warning(
            f"Grouped centroid distance: inconsistent geometry for metal "
            f"{metal_elem} group {names} (radicand R={r_raw:.6f}); "
            f"centroid clamped to the metal (d0=0)."
        )
    return d0, sigma0


def _make_centroid(view_atom, members):
    """Build a serialisable centroid pseudo-atom from a collapsed vertex."""
    return Centroid(view_atom.name, [Atom(member) for member in members])


def _endpoint(view_atom, members):
    """Restraint endpoint for a vertex: Centroid if a group, else a real Atom."""
    if len(members) > 1:
        return _make_centroid(view_atom, members)
    return Atom(members[0])


def _add_grouped_restraints(structure, grouped_view, class_result, clazz_stats):
    """
    Emit the generalised (centroid) restraints for a grouped site:

    * one ``GroupedDistanceStats`` per true group (>= 2 members);
    * one ``GroupedAngleStats`` per vertex pair with at least one group
      (endpoints are polymorphic: Atom for singletons, Centroid for groups);
    * a standard ``AngleStats`` for a singleton-singleton pair.
    """
    ideal_coord = class_result.coord[class_result.index]
    base_ligands = set(structure.ligands)
    view_atoms = list(grouped_view.all_ligands)
    coordination = grouped_view.coordination()

    # metal -> centroid distances (true groups only; singletons live in `base`)
    skipped_groups = set()
    for k, view_atom in enumerate(view_atoms):
        members = grouped_view.group_members(k)
        if len(members) < 2:
            continue
        d0, sigma0 = _grouped_centroid_distance(structure, members)
        if d0 is None:
            skipped_groups.add(k)
            continue
        clazz_stats.add_grouped_bond(
            GroupedDistanceStats(_make_centroid(view_atom, members), d0, sigma0)
        )

    # centroid-M-centroid angles (any pair touching a group -> grouped)
    for i in range(1, coordination):
        for j in range(i + 1, coordination + 1):
            members_i = grouped_view.group_members(i - 1)
            members_j = grouped_view.group_members(j - 1)
            is_group_i = len(members_i) > 1
            is_group_j = len(members_j) > 1

            # a group whose centroid distance was skipped must not leave a
            # dangling angle referencing an undefined centroid
            if (is_group_i and (i - 1) in skipped_groups) or (
                is_group_j and (j - 1) in skipped_groups
            ):
                continue

            ideal_angle = angle(ideal_coord[0], ideal_coord[i], ideal_coord[j])
            if is_group_i or is_group_j:
                clazz_stats.add_grouped_angle(
                    GroupedAngleStats(
                        _endpoint(view_atoms[i - 1], members_i),
                        _endpoint(view_atoms[j - 1], members_j),
                        ideal_angle,
                        5.000,
                    )
                )
            else:
                atom_i = members_i[0]
                atom_j = members_j[0]
                clazz_stats.add_angle(
                    AngleStats(
                        Atom(atom_i),
                        Atom(atom_j),
                        ideal_angle,
                        5.000,
                        is_ligand=atom_i in base_ligands
                        and atom_j in base_ligands,
                    )
                )


def create_grouped_ligand_stats(
    structure: metalCoord.analysis.structures.Ligand,
    class_result: ClassificationResult,
) -> LigandStats:
    grouped_view = get_grouped_view(class_result)
    if grouped_view is None:
        return None

    clazz_stats = LigandStats(
        class_result.clazz,
        create_descriptor(class_result, grouped_view),
        np.array([]),
        class_result.proc,
        grouped_view.coordination(),
        -1,
        f"Grouped fallback ({grouped_view.collapsed_group_count} groups collapsed)",
    )

    # Per-atom ideal d_i fallback lengths for `_chem_comp_bond` connectivity.
    for ligand in structure.ligands:
        clazz_stats.add_bond(_create_fallback_distance_stats(structure, ligand))

    for ligand in structure.extra_ligands:
        clazz_stats.add_pdb_bond(_create_fallback_distance_stats(structure, ligand))

    # Generalised (centroid) geometry -> grouped_base / grouped_angles (JSON).
    _add_grouped_restraints(structure, grouped_view, class_result, clazz_stats)

    return clazz_stats


class StrictCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure, class_result):
        o_ligand_atoms = np.array([structure.metal.atom.name] + structure.atoms())

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

            for file in tqdm(
                files,
                desc=f"{class_result.clazz} ligands",
                leave=False,
                disable=not Logger().progress_bars,
            ):
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                m_ligand_atoms = np.insert(
                    file_data[["Ligand"]].values.ravel(), 0, structure.metal.atom.name
                )

                groups = get_groups(o_ligand_atoms, m_ligand_atoms)

                proc_dists, indices, _, rotateds = fit(
                    pattern_ligand_coord, m_ligand_coord, groups=groups, all=True
                )

                m = n
                for proc_dist, index, rotated in zip(proc_dists, indices, rotateds):
                    if proc_dist >= Config().procrustes_thr():
                        continue

                    sum_coords += rotated[index] - rotated[index][0]
                    n = n + 1

                    procrustes_dists.append(proc_dist)
                    distances.append(
                        np.sqrt(
                            np.sum(
                                (m_ligand_coord[index][0] - m_ligand_coord[index]) ** 2,
                                axis=1,
                            )
                        )[1:].tolist()
                    )
                    angles.append(
                        [
                            angle(
                                m_ligand_coord[index][0],
                                m_ligand_coord[index][i],
                                m_ligand_coord[index][j],
                            )
                            for i in range(1, len(pattern_ligand_coord) - 1)
                            for j in range(i + 1, len(pattern_ligand_coord))
                        ]
                    )

                if m < n:
                    cods[file] = create_gemmi_structure(file_data, rotateds[0])
            procrustes_dists = np.array(procrustes_dists)
            distances = np.array(distances).T
            angles = np.array(angles).T

            if len(distances) > 0 and distances.shape[1] >= Config().min_sample_size:

                clazz_stats = LigandStats(
                    class_result.clazz,
                    create_descriptor(class_result, structure),
                    class_result.lexicographic_order(structure.atom_names_with_symmetries(), structure.element_names()),
                    class_result.proc,
                    structure.coordination(),
                    distances.shape[1],
                    self._finder.description(),
                )
                for file, st in cods.items():
                    clazz_stats.add_cod_file(file, st)

                sum_coords = sum_coords / n

                for i, l in enumerate(list(structure.ligands)):
                    dist, std = modes(distances[i])
                    clazz_stats.add_bond(
                        DistanceStats(
                            Atom(l), dist, std, distances[i], procrustes_dists
                        )
                    )

                for i, l in enumerate(list(structure.extra_ligands)):
                    dist, std = modes(distances[i + structure.ligands_len])
                    clazz_stats.add_pdb_bond(
                        DistanceStats(
                            Atom(l),
                            dist,
                            std,
                            euclidean(
                                sum_coords[i + 1 + structure.ligands_len], sum_coords[0]
                            ),
                            procrustes_dists,
                        )
                    )

                if Config().ideal_angles:
                    self.add_ideal_angels(structure, class_result, clazz_stats)
                else:
                    k = 0
                    n1 = structure.ligands_len
                    ligands = list(structure.all_ligands)
                    for i in range(structure.coordination() - 1):
                        for j in range(i + 1, structure.coordination()):
                            a, std = calculate_stats(angles[k])
                            clazz_stats.add_angle(
                                AngleStats(
                                    Atom(ligands[i]),
                                    Atom(ligands[j]),
                                    a,
                                    std,
                                    is_ligand=i < n1 and j < n1,
                                    angles=angles[k],
                                    procrustes_dists=procrustes_dists,
                                )
                            )
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
            for file in tqdm(
                files,
                desc=f"{class_result.clazz} ligands",
                leave=False,
                disable=not Logger().progress_bars,
            ):
                file_data = self._finder.data(file)

                m_ligand_coord = get_coordinate(file_data)
                proc_dist, _, _, r, _ = fit(pattern_ligand_coord, m_ligand_coord)

                if proc_dist < Config().procrustes_thr():
                    distances.append(
                        np.sqrt(
                            np.sum((m_ligand_coord[0] - m_ligand_coord) ** 2, axis=1)
                        )[1:].tolist()
                    )
                    lig_names.append(file_data[["Ligand"]].values.ravel().tolist())
                    cods[file] = create_gemmi_structure(file_data, m_ligand_coord @ r)

            distances = np.array(distances).T
            lig_names = np.array(lig_names).T

            if len(distances) > 0 and distances.shape[1] >= Config().min_sample_size:
                clazz_stats = LigandStats(
                    class_result.clazz,
                    create_descriptor(class_result, structure),
                    class_result.lexicographic_order(structure.atom_names_with_symmetries(), structure.element_names()),
                    class_result.proc,
                    structure.coordination(),
                    distances.shape[1],
                    self._finder.description(),
                )
                for file, st in cods.items():
                    clazz_stats.add_cod_file(file, st)

                ligands = list(structure.ligands)

                results = {}
                for element in np.unique(lig_names):
                    element_distances = distances.ravel()[lig_names.ravel() == element]

                    if element_distances.size == 1:
                        results[element] = modes(element_distances)
                    elif element_distances.size > 1:
                        results[element] = modes(element_distances.squeeze())

                for l in ligands:
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazz_stats.add_bond(DistanceStats(Atom(l), dist, std))
                    else:
                        clazz_stats.add_bond(
                            self._create_covalent_distance_stats(
                                structure, l, "Covalent distance"
                            )
                        )

                for l in structure.extra_ligands:
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazz_stats.add_pdb_bond(DistanceStats(Atom(l), dist, std))
                    else:
                        clazz_stats.add_pdb_bond(
                            self._create_covalent_distance_stats(
                                structure, l, "Covalent distance"
                            )
                        )

                self.add_ideal_angels(structure, class_result, clazz_stats)

                return clazz_stats
        return None


class OnlyDistanceStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        self._finder.load(structure, data)
        data = self._finder.data("")
        clazz_stats = LigandStats(
            class_result.clazz,
            create_descriptor(class_result, structure),
            class_result.lexicographic_order(structure.atom_names_with_symmetries(), structure.element_names()),
            class_result.proc,
            structure.coordination(),
            -1,
            self._finder.description(),
        )

        for l in structure.ligands:
            dist, std, count = DB.get_distance_stats(
                structure.metal.atom.element.name, l.atom.element.name
            )
            if count > 0:
                clazz_stats.add_bond(
                    DistanceStats(Atom(l), np.array([dist]), np.array([std]))
                )
            else:
                clazz_stats.add_bond(
                    self._create_covalent_distance_stats(
                        structure, l, "Covalent distance"
                    )
                )

        for l in structure.extra_ligands:
            dist, std, count = DB.get_distance_stats(
                structure.metal.atom.element.name, l.atom.element.name
            )
            if count > 0:
                clazz_stats.add_pdb_bond(
                    DistanceStats(Atom(l), np.array([dist]), np.array([std]))
                )
            else:
                clazz_stats.add_pdb_bond(
                    self._create_covalent_distance_stats(
                        structure, l, "Covalent distance"
                    )
                )

        self.add_ideal_angels(structure, class_result, clazz_stats)

        if clazz_stats.bond_count > 0:
            return clazz_stats
        return None


class CovalentStatsFinder(StatsFinder):

    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data, class_result):
        clazz_stats = LigandStats(
            class_result.clazz if class_result else "",
            create_descriptor(class_result, structure) if class_result else "",
            class_result.lexicographic_order(structure.atom_names_with_symmetries(), structure.element_names()) if class_result else [],
            class_result.proc if class_result else -1,
            structure.coordination(),
            -1,
            self._finder.description(),
        )

        for l in structure.ligands:
            clazz_stats.add_bond(self._create_covalent_distance_stats(structure, l))

        for l in structure.extra_ligands:
            clazz_stats.add_pdb_bond(self._create_covalent_distance_stats(structure, l))

        self.add_ideal_angels(structure, class_result, clazz_stats)

        return clazz_stats
