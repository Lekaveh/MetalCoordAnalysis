from abc import ABC, abstractmethod
from itertools import chain
from arrow import get
import numpy as np
import gemmi
from metalCoord.config import Config
from metalCoord.logging import Logger
from metalCoord.cif.utils import ATOM_ID, TYPE_SYMBOL, COORDS, ATOM_ID_1


class IAtom(ABC):
    """Abstract base class representing an atom in a molecular structure."""

    @property
    @abstractmethod
    def atom(self):
        """Get the atom identifier."""

    @property
    @abstractmethod
    def residue(self):
        """Get the residue the atom belongs to."""

    @property
    @abstractmethod
    def chain(self):
        """Get the chain the atom belongs to."""

    @property
    @abstractmethod
    def symmetry(self):
        """Get if the atom comes from the symmetry."""

    @property
    def pos(self):
        """Get the position of the atom."""


class Atom(IAtom):
    """
    Represents an atom in a molecular structure.

    Attributes:
        atom (gemmi.Atom): The atom.
        residue (gemmi.Residue): The name of the residue the atom belongs to.
        chain (gemi.Chain): The chain identifier of the atom.
        mark (gemmi.NeighborSearch.Mark): The mark associated with the atom.
        st (gemmi.Structure): The structure the atom belongs to.
        metal (gemmi.Atom, optional): The metal atom associated with this atom, if any.
    """

    def __init__(self, atom: gemmi.Atom, residue: gemmi.Residue, chain: gemmi.Chain, mark: gemmi.NeighborSearch.Mark, st: gemmi.Structure, metal: gemmi.Atom = None):
        self._atom = atom
        self._residue = residue
        self._chain = chain
        self._mark = mark
        self._st = st
        self._metal = metal

    @property
    def atom(self):
        """
        Get the atom associated with this structure.

        Returns:
            The atom object associated with this structure.
        """
        return self._atom

    @property
    def name(self):
        """
        Get the name of the atom.

        Returns:
            The name of the atom.
        """
        return self._atom.name

    @property
    def element(self):
        """
        Get the element of the atom.

        Returns:
            The element of the atom.
        """
        return self._atom.element.name

    @property
    def residue(self):
        """
        Returns the residue associated with the structure.

        Returns:
            The residue object associated with the structure.
        """
        return self._residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain identifier.
        """
        return self._chain

    @property
    def symmetry(self):
        """
        Returns if the atom comes from the symmetry.

        Returns:
            bool: The symmetry copy of the atom.
        """
        return self._mark.image_idx

    @property
    def pos(self):
        """
        Returns the position of the atom.

        Returns:if
            np.array: The position of the atom.
        """

        if self._mark and self._metal:
            if self.symmetry:
                return self._st.cell.find_nearest_pbc_position(self._metal.pos, self._mark.pos, 0)
        return self._atom.pos


class CifAtom(IAtom):
    """
     Represents an atom in a molecular structure.

    Attributes:
        atom (gemmi.Atom): The atom object.
        residue (gemmi.Residue): The residue the atom belongs to.
        chain (gemmi.Chain): The chain the atom belongs to.
    """

    def __init__(self, atom: gemmi.Atom, residue: gemmi.Residue, chain: gemmi.Chain):
        self._atom = atom
        self._residue = residue
        self._chain = chain

    @property
    def atom(self):
        """
        Get the atom associated with this structure.

        Returns:
            The atom object associated with this structure.
        """
        return self._atom

    @property
    def residue(self):
        """
        Returns the residue associated with the structure.

        Returns:
            The residue object associated with the structure.
        """
        return self._residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain identifier.
        """
        return self._chain

    @property
    def symmetry(self):
        """
        Returns if the atom comes from the symmetry.

        Returns:
            bool: The symmetry copy of the atom.
        """
        return 0

    @property
    def pos(self):
        """
        Returns the position of the atom.

        Returns:
            np.array: The position of the atom.
        """
        return self._atom.pos

    @property
    def name(self):
        """
        Get the name of the atom.

        Returns:
            The name of the atom.
        """
        return self._atom.name

    @property
    def element(self):
        """
        Get the element of the atom.

        Returns:
            The element of the atom.
        """
        return self._atom.element.name


class Ligand:
    """ The `Ligand` class represents a metal coordination structure with ligands and extra ligands.
    Attributes:
        _metal (IAtom): The metal atom associated with the structure.
        _ligands (list): A list of ligand atoms.
        _extra_ligands (list): A list of extra ligand atoms.
    """

    def __init__(self, metal: IAtom) -> None:
        self._metal = metal
        self._ligands = []
        self._extra_ligands = []

    def clean_the_farthest(self, free: bool = False, n: int = 1) -> "Ligand":
        """
        Cleans the n farthest ligands from the metal coordination.

        Args:
            free (bool, optional): If True, includes ligands from both self._ligands and self._extra_ligands.
                                If False, includes only ligands from self._extra_ligands. Default is False.
            n (int, optional): Number of farthest ligands to remove. Default is 1.

        Returns
            Ligand: A new Ligand object with the n farthest ligands removed.

        """
        atoms = self._ligands + self._extra_ligands if free else self._extra_ligands
        # Sort the ligands by distance and get the atoms of the n farthest
        to_delete_atoms = [l[0] for l in sorted([[l, self._cov_dist_coeff(l)]
                                                for l in atoms], key=lambda x: x[1], reverse=True)[:n]]
        cleaned_ligand = Ligand(self._metal)

        for l in self._ligands:
            if free:
                if l not in to_delete_atoms:
                    cleaned_ligand.add_ligand(l)
            else:
                cleaned_ligand.add_ligand(l)
        for l in self._extra_ligands:
            if l not in to_delete_atoms:
                cleaned_ligand.add_extra_ligand(l)

        return cleaned_ligand

    def _euclidean(self, atom1: IAtom, atom2: IAtom) -> float:
        """
        Calculates the Euclidean distance between two atoms.

        Args:
            atom1 (IAtom): The first atom.
            atom2 (IAtom): The second atom.

        Returns:
            The Euclidean distance between the two atoms.
        """
        return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

    def _dist(self, l: IAtom):
        """
        Calculates the distance between the metal and a ligand atom.

        Args:
            l (IAtom): The ligand atom.

        Returns:
            The distance between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l)

    def _cov_dist(self, l: IAtom):
        """
        Calculates the covalent distance between the metal and a ligand atom.

        Args:
            l (IAtom): The ligand atom.

        Returns:
            The covalent distance between the metal and the ligand atom.
        """
        return gemmi.Element(self.metal.atom.element.name).covalent_r + gemmi.Element(l.atom.element.name).covalent_r

    def _cov_dist_coeff(self, l: IAtom):
        """
        Calculates the covalent distance coefficient between the metal and a ligand atom.

        Args:
            l (IAtom): The ligand atom.

        Returns:
            The covalent distance coefficient between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l) / self._cov_dist(l)

    def cov_dist_coeffs(self):
        """
        Calculates the covalent distance coefficients for all extra ligands.

        Returns:
            A dictionary containing the covalent distance coefficients for each extra ligand.
        """
        return {l.atom.name: self._cov_dist_coeff(l) for l in self._extra_ligands}

    @property
    def metal(self):
        """
        Returns the metal associated with the structure.

        Returns:
            The metal element.
        """
        return self._metal

    @property
    def residue(self):
        """
        Returns the residue associated with the structure.

        Returns:
            The residue object associated with the structure.
        """
        return self._metal.residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain associated with the structure.
        """
        return self._metal.chain

    @property
    def ligands(self):
        """
        Generator that yields each ligand associated with the metal.

        Yields:
            Each ligand atom.
        """
        for ligand in self._ligands:
            yield ligand

    @property
    def extra_ligands(self):
        """
        Generator that yields each extra ligand associated with the metal.

        Yields:
            Each extra ligand atom.
        """
        for ligand in self._extra_ligands:
            yield ligand

    @property
    def all_ligands(self):
        """
        Generator that yields each ligand and extra ligand associated with the metal.

        Yields:
            Each ligand and extra ligand atom.
        """
        for ligand in self._ligands + self._extra_ligands:
            yield ligand

    @property
    def ligands_len(self):
        """
        Returns the number of ligands associated with the metal.

        Returns:
            The number of ligands.
        """
        return len(self._ligands)

    def add_ligand(self, ligand: IAtom):
        """
        Adds a ligand to the structure.

        Args:
            ligand (IAtom): The ligand to add.
        """

        # if self.metal.atom.occ < 1 and ligand.symmetry and (ligand.atom.occ + self.metal.atom.occ) <= 1.0:
        #     return

        self._ligands.append(ligand)

    def add_extra_ligand(self, ligand: IAtom):
        """
        Adds an extra ligand to the structure.

        Args:
            ligand (IAtom): The extra ligand to add.
        """

        self._extra_ligands.append(ligand)

    def elements(self):
        """
        Returns a sorted list of element names for all ligands and extra ligands.

        Returns:
            A sorted list of element names.
        """
        return sorted([ligand.atom.element.name for ligand in self._ligands + self._extra_ligands])

    def names(self):
        """
        Returns a sorted list of names for all ligands and extra ligands.

        Returns:
            A sorted list of  names.
        """
        return sorted([ligand.atom.name for ligand in self._ligands + self._extra_ligands])

    def symmetry_names(self):
        """
        Returns a sorted list of names with symmetries for all ligands and extra ligands.

        Returns:
            A sorted list of  names.
        """
        return sorted([f"{ligand.name} {'(' + str(ligand.symmetry) + ')' if ligand.symmetry else ''}" for ligand in self._ligands + self._extra_ligands])

    def atoms(self):
        """
        Returns a list of element names for all ligands and extra ligands.

        Returns:
            A list of element names.
        """
        return [ligand.atom.element.name for ligand in self._ligands + self._extra_ligands]

    def code(self):
        """
        Generates a code representation of the ligand.

        Returns:
            The code representation of the ligand.
        """
        return "".join([self._metal.atom.element.name] + self.elements())

    def name_code(self):
        """
        Returns the name code of the metal coordination structure.

        The name code is generated by concatenating the name of the metal with the names of all the atoms in the structure.

        Returns:
            str: The name code of the metal coordination structure.
        """
        return " ".join([self._metal.atom.name] + self.names())

    def name_code_with_symmetries(self):
        """
        Returns the name code of the metal coordination structure.

        The name code is generated by concatenating the name of the metal with the names of all the atoms in the structure.

        Returns:
            str: The name code of the metal coordination structure.
        """

        return " ".join([self._metal.atom.name] + self.symmetry_names())

    def get_coord(self):
        """
        Returns the coordinates of the metal and ligand atoms.

        Returns:
            An array containing the coordinates of the metal and ligand atoms.
        """
        return np.array([self._metal.pos.tolist()] + [ligand.pos.tolist() for ligand in self._ligands + self._extra_ligands])

    def coordination(self):
        """
        Returns the coordination number of the metal.

        Returns:
            The coordination number.
        """
        return len(self._ligands) + len(self._extra_ligands)

    def contains(self, atom):
        """
        Checks if the ligand contains a specific atom.

        Args:
            atom (gemmi.IAtom): The atom to check.

        Returns:
            True if the ligand contains the atom, False otherwise.
        """
        return atom in [ligand.atom for ligand in self._ligands + self._extra_ligands]

    def filter_extra(self):
        """
        Filters out extra ligands that are too close to any ligand atom.
        """
        to_delete = []
        for i, atom1 in enumerate(self._extra_ligands):
            for atom2 in self._ligands:
                if distance(atom1, atom2) < 0.5:
                    to_delete.append(i)
                    break
        if to_delete:
            self._extra_ligands = [atom for i, atom in enumerate(
                self._extra_ligands) if i not in to_delete]

    def filter_base(self):
        """
        Filters out ligands that are too far from the metal based on covalent distance coefficient.
        """
        to_delete = []
        for i, atom1 in enumerate(self._ligands):
            if self._cov_dist_coeff(atom1) > 2:
                to_delete.append(i)

        if to_delete:
            self._ligands = [atom for i, atom in enumerate(
                self._ligands) if i not in to_delete]

    def mean_occ(self):
        """
        Calculates the mean occupancy of the metal and ligand atoms.

        Returns:
            The mean occupancy.
        """
        return np.mean([self._metal.atom.occ] + [ligand.atom.occ for ligand in self._ligands + self._extra_ligands])

    def mean_b(self):
        """
        Calculates the mean B-factor of the metal and ligand atoms.

        Returns:
            The mean B-factor.
        """
        return np.mean([self._metal.atom.b_iso] + [ligand.atom.b_iso for ligand in self._ligands + self._extra_ligands])

    def __str__(self) -> str:
        return f"{self._metal.atom.name} - {self.chain.name} - {self.residue.name} - {self.residue.seqid.num}"

    def __repr__(self) -> str:
        ligands = " ".join([ligand.atom.name for ligand in self._ligands])
        extra_ligands = " ".join(
            [ligand.atom.name for ligand in self._extra_ligands])
        return f"{self._metal.atom.name} - {self.chain.name} - {self.chain.name} - {self.residue.seqid.num} - {ligands} - {extra_ligands}"


def angle(atom1: IAtom, atom2: IAtom, atom3: IAtom):
    """
    Calculates the angle between three atoms.

    Args:
        atom1 (IAtom): The first atom.
        atom2 (IAtom): The second atom.
        atom3 (IAtom): The third atom.

    Returns:
        The angle in degrees between the three atoms.
    """
    vec1 = atom1.pos - atom2.pos
    vec2 = atom3.pos - atom2.pos
    cosine_angle = vec1.dot(
        vec2) / (np.linalg.norm([vec1.x, vec1.y, vec1.z]) * np.linalg.norm([vec2.x, vec2.y, vec2.z]))
    return np.degrees(np.arccos(cosine_angle))


def distance(atom1: IAtom, atom2: IAtom) -> float:
    """
    Calculates the Euclidean distance between two atoms.

    Args:
        atom1 (IAtom): The first atom.
        atom2 (IAtom): The second atom.

    Returns:
        The Euclidean distance between the two atoms.
    """
    return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))


def get_ligands(st, ligand, bonds=None, max_dist=10, only_best=False) -> list[Ligand]:
    """
    Retrieves ligands associated with a metal in a structure.

    Args:
        st (gemmi.Structure): The structure.
        ligand (str): The name of the ligand.
        bonds (dict): Dictionary of metal-ligand bond information.
        max_dist (float): The maximum distance for ligand searching.
        only_best (bool): Whether to return only the best ligand structures.

    Returns:
        A list of Ligand objects representing the ligands associated with the metal.
    """
    scale = Config().scale()
    alpha = Config().distance_threshold + 1
    beta1 = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    beta1 = [b for b in beta1 if b < alpha]
    alpha1 = 1.1
    angle1 = 60

    def covalent_radii(element):
        return gemmi.Element(element).covalent_r

    def find_min_angle_and_update(center, n0, n1, beta_c):
        if not n1:
            return n0, n1

        while True:
            l_n1 = len(n1)
            angles = [
                (angle(a1, center, a2), a1, a2)
                for a1 in n1 + n0 for a2 in n1 + n0 if a1 != a2
            ]

            if all(a > angle1 for a, _, _ in angles):
                return n0, n1

            angles = list(
                filter(lambda x: (x[1] in n1) or (x[2] in n1), angles))
            _, min_a1, min_a2 = min(angles, key=lambda x: x[0])

            coef_i = distance(center, min_a1) / (covalent_radii(
                center.atom.element.name) + covalent_radii(min_a1.atom.element.name))
            coef_j = distance(center, min_a2) / (covalent_radii(
                center.atom.element.name) + covalent_radii(min_a2.atom.element.name))
            max_coef_atom = min_a1 if coef_i > coef_j else min_a2

            if max_coef_atom in n1 and max(coef_i, coef_j) > beta_c:
                n1.remove(max_coef_atom)

            if len(n1) == l_n1:
                return n0, n1

            if not n1:
                return n0, n1

    if st is None:
        return None

    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate(include_h=False)
    structures = []

    if not bonds:
        bonds = {}

    for chain in st[0]:
        for residue in chain:
            if residue.name != ligand:
                continue

            for atom in residue:
                if atom.element.is_metal:
                    metal_name = atom.name
                    metal_bonds = set(bonds.get(metal_name, []))
                    metal = Atom(atom, residue, chain, None, st)
                    ligand_obj = Ligand(metal)

                    marks = ns.find_neighbors(
                        atom, min_dist=0.1, max_dist=max_dist)

                    marks1 = [k for k in marks if k.image_idx ==
                              0 and not k.to_cra(st[0]).atom.element.is_metal]
                    marks2 = [k for k in marks if k.image_idx !=
                              0 and not k.to_cra(st[0]).atom.element.is_metal]
                    for mark2 in marks2:
                        cra2 = mark2.to_cra(st[0])
                        if cra2.atom.occ + atom.occ > 1:
                            marks1.append(mark2)
                            continue
                        for mark1 in marks1:
                            cra1 = mark1.to_cra(st[0])
                            if (cra1.atom.name == cra2.atom.name and cra1.residue.seqid.num == cra2.residue.seqid.num and
                                    cra1.chain.name == cra2.chain.name and cra1.atom.occ + cra2.atom.occ > atom.occ):
                                break
                        else:
                            marks1.append(mark2)
                    marks = marks1
                    if Config().simple:
                        for mark in marks:
                            cra = mark.to_cra(st[0])
                            if cra.atom.element.is_metal:
                                continue

                            l = Atom(cra.atom, cra.residue,
                                     cra.chain, mark, st, atom)
                            if bonds:
                                if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name and a.symmetry == 0:
                                    if cra.atom.name in metal_bonds:
                                        ligand_obj.add_ligand(l)
                                elif distance(metal, l) <= (covalent_radii(metal.atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                    ligand_obj.add_extra_ligand(l)
                            elif distance(metal, l) <= (covalent_radii(metal.atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name:
                                    ligand_obj.add_ligand(l)
                                else:
                                    ligand_obj.add_extra_ligand(l)
                    else:
                        k = len(beta1) - 1
                        # Step 1: Select all atoms for which d(m, i) < alpha * (r_m + r_i). Denote this set as n1.

                        neighbour_atoms = []
                        for mark in marks:
                            cra = mark.to_cra(st[0])
                            neighbour_atoms.append(
                                Atom(cra.atom, cra.residue, cra.chain, mark, st, atom))

                        ligand_atoms = []
                        if bonds:
                            ligand_atoms = [a for a in neighbour_atoms if a.atom.name in metal_bonds and a.residue.name ==
                                            ligand and a.residue.seqid.num == residue.seqid.num and a.chain.name == chain.name and a.symmetry == 0]
                            neighbour_atoms = [
                                a for a in neighbour_atoms if a not in ligand_atoms]

                        n1 = [
                            neighbour_atom for neighbour_atom in neighbour_atoms
                            if not neighbour_atom.atom.element.is_metal and distance(metal, neighbour_atom) < (covalent_radii(metal.atom.element.name) + covalent_radii(neighbour_atom.atom.element.name)) * alpha
                            and distance(metal, neighbour_atom) > (covalent_radii(metal.atom.element.name) + covalent_radii(neighbour_atom.atom.element.name)) * alpha1
                        ]

                        # Step 2: Select all atoms for which d(m, i) <= alpha1 * (r_m + r_i). Denote this set n0.
                        n0 = [
                            neighbour_atom for neighbour_atom in neighbour_atoms
                            if not neighbour_atom.atom.element.is_metal and distance(metal, neighbour_atom) <= (covalent_radii(metal.atom.element.name) + covalent_radii(neighbour_atom.atom.element.name)) * alpha1
                        ] + ligand_atoms

                        # Step 3: Remove atoms in n0 from n1
                        n1 = [a for a in n1 if a not in n0]

                        # Step 4-9: Apply the logic iteratively
                        beta_c = beta1[k]
                        while k >= 0:
                            n0, n1 = find_min_angle_and_update(
                                ligand_obj.metal, n0, n1, beta_c)
                            k -= 1
                            beta_c = beta1[k]

                        n0.extend(n1)

                        # Add atoms to ligand_obj
                        neighbour_atoms = neighbour_atoms + ligand_atoms
                        for a in n0:
                            idx = [i for i, x in enumerate(
                                neighbour_atoms) if x == a][0]
                            mark = marks[idx]
                            if mark.image_idx:
                                Logger().info(
                                    f"This atom {a.atom.name} in {chain.name} - {residue.name} - {residue.seqid.num} come from symmetry copy {mark.image_idx}")

                            if a.residue.name == ligand and a.residue.seqid.num == residue.seqid.num and a.chain.name == chain.name:
                                ligand_obj.add_ligand(a)

                            else:
                                ligand_obj.add_extra_ligand(a)

                    # ligand_obj.filter_base()
                    ligand_obj.filter_extra()

                    if Config().max_coordination_number and ligand_obj.coordination() > Config().max_coordination_number:
                        ligand_obj = ligand_obj.clean_the_farthest(
                            free=bool(bonds), n=ligand_obj.coordination() - Config().max_coordination_number)

                    if bonds and ligand_obj.ligands_len != len(bonds.get(metal_name, [])):
                        raise ValueError(
                            f"There is inconsistency between ligand(s) in the PDB and monomer file. Metal {metal_name} in {chain.name} - {residue.name} - {residue.seqid.num} has different number of neighbours than expected. Expected: {sorted(bonds.get(metal_name, []))}, found: {sorted([l.atom.name for l in ligand_obj.ligands])}")

                    structures.append(ligand_obj)

    if only_best:
        best_structures = []
        metals = np.unique(
            [structure.metal.atom.name for structure in structures]).tolist()

        for metal_name in metals:
            metal_stuctures = [
                s for s in structures if metal_name == s.metal.name]
            b_vlaues = [metal.mean_b() for metal in metal_stuctures]
            occ_values = [metal.mean_occ() for metal in metal_stuctures]
            best_occ = np.max(occ_values)
            best_structures.append(metal_stuctures[np.argmin(
                np.where(occ_values == best_occ, b_vlaues, np.inf))])
        structures = best_structures

    return structures


def is_float(value: str) -> bool:
    """
    Checks if the given string can be converted to a float.

    Args:
        value (str): The string to check.

    Returns:
        bool: True if the string can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:

        return False


def get_coord_sym(atoms: gemmi.cif.Table) -> list:
    """
    Retrieve the coordinate symbols from a CIF table.

    Args:
        atoms (gemmi.cif.Table): The CIF table containing atomic data.

    Returns:
        list: The coordinate symbols.
    """
    for coord_sym in COORDS:
        if coord_sym[0] in atoms and all([is_float(x) & is_float(y)
                                          & is_float(z) for x, y, z in zip(atoms[coord_sym[0]], atoms[coord_sym[1]], atoms[coord_sym[2]])]):
            return coord_sym
    return None

def get_row(atoms: gemmi.cif.Table, metal: str):
    """
    Retrieve a specific row from a CIF table based on a metal identifier.

    Args:
        atoms (gemmi.cif.Table): The CIF table containing atomic data.
        metal (str): The identifier of the metal atom to search for.

    Returns:
        gemmi.cif.Row: The row corresponding to the specified metal atom, or None if not found.
    """


    coord_syms = get_coord_sym(atoms)
    if not coord_syms:
        raise ValueError("Could not find coordinates in the CIF file.")


    for atom, element, x, y, z in zip(atoms[ATOM_ID], atoms[TYPE_SYMBOL], atoms[coord_syms[0]], atoms[coord_syms[1]], atoms[coord_syms[2]]):
        if atom == metal:
            return atom, element, float(x), float(y), float(z)

def create_atom(atoms, atom_name):
    """
    Create a gemmi.Atom object from the given atoms data and atom name.

    Args:
        atoms (list): A list of atom data.
        atom_name (str): The name of the atom to create.

    Returns:
        gemmi.Atom: A gemmi.Atom object with the specified properties.
    """

    name, element, x, y, z = get_row(atoms, atom_name)
    ligand = gemmi.Atom()
    ligand.name = name
    ligand.element = gemmi.Element(element)
    ligand.pos =  gemmi.Position(x, y, z)
    ligand.occ = 1
    ligand.b_iso = 0
    ligand.altloc = "."
    return ligand

def get_ligands_from_cif(name: str, atoms: gemmi.cif.Table, bonds: dict) -> list[Ligand]:
    """
    Extracts ligands from a CIF (Crystallographic Information File) and returns them as a list of Ligand objects.
    Args:
        name (str): The name of the residue.
        atoms (gemmi.cif.Table): A table containing atom information from the CIF file.
        bonds (dict): A dictionary where keys are metal atom names and values are lists of ligand atom names bonded to the metal.
    Returns:
        list[Ligand]: A list of Ligand objects representing the ligands bonded to the metals.
    """
    result = []
    new_chain = gemmi.Chain("A")
    seq_id = gemmi.SeqId("1")
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = seq_id
    for metal_name, ligands in bonds.items():
        metal = create_atom(atoms, metal_name)
        ligand_obj = Ligand(CifAtom(metal, residue, new_chain))
        for ligand_name in ligands:
            ligand = create_atom(atoms, ligand_name)
            ligand_obj.add_ligand(CifAtom(ligand, residue, new_chain))

        result.append(ligand_obj)

    return result
