import numpy as np
import gemmi
from metalCoord.config import Config


class Atom:
    """
    Represents an atom in a molecular structure.

    Attributes:
        atom (str): The name of the atom.
        residue (str): The name of the residue the atom belongs to.
        chain (str): The chain identifier of the atom.
    """

    def __init__(self, atom, residue, chain):
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


class Ligand:
    """
    Represents a ligand associated with a metal in a structure.

    Attributes:
        _metal (gemmi.Atom): The metal atom.
        _residue (gemmi.Residue): The residue containing the metal.
        _chain (gemmi.Chain): The chain containing the residue.
        _ligands (list): List of atoms from ligand associated with the metal.
        _extra_ligands (list): List of additional atom not from ligands associated with the metal.
    """

    def __init__(self, metal, residue, chain) -> None:
        self._metal = metal
        self._residue = residue
        self._chain = chain
        self._ligands = []
        self._extra_ligands = []

    def _euclidean(self, atom1: gemmi.Atom, atom2: gemmi.Atom):
        """
        Calculates the Euclidean distance between two atoms.

        Args:
            atom1 (gemmi.Atom): The first atom.
            atom2 (gemmi.Atom): The second atom.

        Returns:
            The Euclidean distance between the two atoms.
        """
        return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

    def _dist(self, l: Atom):
        """
        Calculates the distance between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The distance between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l.atom)

    def _cov_dist(self, l: Atom):
        """
        Calculates the covalent distance between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The covalent distance between the metal and the ligand atom.
        """
        return gemmi.Element(self.metal.element.name).covalent_r + gemmi.Element(l.atom.element.name).covalent_r

    def _cov_dist_coeff(self, l: Atom):
        """
        Calculates the covalent distance coefficient between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The covalent distance coefficient between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l.atom) / self._cov_dist(l)

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
        return self._residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain associated with the structure.
        """
        return self._chain

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

    def add_ligand(self, ligand: Atom):
        """
        Adds a ligand to the structure.

        Args:
            ligand (Atom): The ligand to add.
        """
        self._ligands.append(ligand)

    def add_extra_ligand(self, ligand: Atom):
        """
        Adds an extra ligand to the structure.

        Args:
            ligand (Atom): The extra ligand to add.
        """
        self._extra_ligands.append(ligand)

    def elements(self):
        """
        Returns a sorted list of element names for all ligands and extra ligands.

        Returns:
            A sorted list of element names.
        """
        return sorted([ligand.atom.element.name for ligand in self._ligands + self._extra_ligands])

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
        return "".join([self._metal.element.name] + self.elements())
    


    def get_coord(self):
        """
        Returns the coordinates of the metal and ligand atoms.

        Returns:
            An array containing the coordinates of the metal and ligand atoms.
        """
        return np.array([self._metal.pos.tolist()] + [ligand.atom.pos.tolist() for ligand in self._ligands + self._extra_ligands])

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
            atom (gemmi.Atom): The atom to check.

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
                if distance(atom1.atom, atom2.atom) < 1:
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
        return np.mean([self._metal.occ] + [ligand.atom.occ for ligand in self._ligands + self._extra_ligands])

    def mean_b(self):
        """
        Calculates the mean B-factor of the metal and ligand atoms.

        Returns:
            The mean B-factor.
        """
        return np.mean([self._metal.b_iso] + [ligand.atom.b_iso for ligand in self._ligands + self._extra_ligands])

    def __str__(self) -> str:
        return f"{self._metal.name} - {self._chain.name} - {self._residue.name} - {self._residue.seqid.num}"

    def __repr__(self) -> str:
        ligands = " ".join([ligand.atom.name for ligand in self._ligands])
        extra_ligands = " ".join([ligand.atom.name for ligand in self._extra_ligands])
        return f"{self._metal.name} - {self._chain.name} - {self._residue.name} - {self._residue.seqid.num} - {ligands} - {extra_ligands}"

def distance(atom1, atom2):
    """
    Calculates the Euclidean distance between two atoms.

    Args:
        atom1 (gemmi.Atom): The first atom.
        atom2 (gemmi.Atom): The second atom.

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

    def covalent_radii(element):
        return gemmi.Element(element).covalent_r

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
                    ligand_obj = Ligand(atom, residue, chain)
                    structures.append(ligand_obj)

                    marks = ns.find_neighbors(
                        atom, min_dist=0.1, max_dist=max_dist)
                    for mark in marks:
                        cra = mark.to_cra(st[0])
                        if cra.atom.element.is_metal:
                            continue
                        if ligand_obj.contains(cra.atom):
                            continue

                        if bonds:

                            if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name:
                                # if cra.residue.name == ligand:
                                if cra.atom.name in metal_bonds:
                                    ligand_obj.add_ligand(
                                        Atom(cra.atom, cra.residue, cra.chain))
                            elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                ligand_obj.add_extra_ligand(
                                    Atom(cra.atom, cra.residue, cra.chain))
                        elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                            if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name:
                                ligand_obj.add_ligand(
                                    Atom(cra.atom, cra.residue, cra.chain))
                            else:
                                ligand_obj.add_extra_ligand(
                                    Atom(cra.atom, cra.residue, cra.chain))

                    # ligand_obj.filter_base()
                    ligand_obj.filter_extra()

                    if ligand_obj.ligands_len < len(bonds.get(metal_name, [])):
                        raise ValueError(
                            f"There is inconsistency between ligand(s) in the PDB and monomer file. Metal {metal_name} in {chain.name} - {residue.name} - {residue.seqid.num} has fewer neighbours than expected. Expected: {sorted(bonds.get(metal_name, []))}, found: {sorted([l.atom.name for l in ligand_obj.ligands])}")

    if only_best:
        best_structures = []
        metals = np.unique(
            [structure.metal.name for structure in structures]).tolist()

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
