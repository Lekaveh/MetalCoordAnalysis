import numpy as np

class Ligand():
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

    def __init__(self, ligand) -> None:
        self._name = ligand.atom.name
        self._element = ligand.atom.element.name
        self._chain = ligand.chain.name
        self._residue = ligand.residue.name
        self._sequence = ligand.residue.seqid.num
        self._icode = ligand.residue.seqid.icode.strip()
        self._altloc = ligand.atom.altloc

    @property
    def name(self):
        """
        Returns the name of the object.

        Returns:
            str: The name of the object.
        """
        return self._name

    @property
    def element(self):
        """
        Returns the element associated with this object.

        Returns:
            The element associated with this object.
        """
        return self._element

    @property
    def chain(self):
        """
        Returns the chain associated with the object.

        Returns:
            The chain associated with the object.
        """
        return self._chain

    @property
    def residue(self):
        """
        Returns the residue associated with the object.

        Returns:
            The residue associated with the object.
        """
        return self._residue

    @property
    def sequence(self):
        """
        Returns the sequence associated with the object.

        Returns:
            str: The sequence associated with the object.
        """
        return self._sequence

    @property
    def insertion_code(self):
        """
        Returns the insertion code of the atom.
        

        Returns:
            str: The insertion code of the atom.
        """
        if self._icode == "\u0000":
            return "."
        return self._icode if self._icode else "."

    def equals(self, other):
        """
        Check if the current object is equal to another object.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        return self.code == other.code

    @property
    def altloc(self):
        """
        Returns the alternative location indicator for the atom.

        If the alternative location indicator is the null character ('\u0000'),
        an empty string is returned.

        Returns:
            str: The alternative location indicator for the atom.
        """
        return "" if self._altloc == "\u0000" else self._altloc

    @property
    def code(self):
        """
        Returns a tuple containing the name, element, chain, residue, sequence, and altloc of the object.

        Returns:
            tuple: A tuple containing the name, element, chain, residue, sequence, and altloc.
        """
        return (self.name, self.element, self.chain, self.residue, self.sequence, self.altloc)

    def to_dict(self):
        """
        Converts the object to a dictionary.

        Returns:
            dict: A dictionary representation of the object.
        """
        return {"name": self.name, "element": self.element, "chain": self.chain, "residue": self.residue, "sequence": self.sequence, "icode": self.insertion_code, "altloc": self.altloc}


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

    def __init__(self, ligand1, ligand2, angle_value, std, is_ligand=True, angles=None, procrustes_dists=None) -> None:
        """
        Initialize an AngleStats object.

        Args:
            ligand1 (Ligand): The first ligand.
            ligand2 (Ligand): The second ligand.
            angle_value (float): The angle value.
            std (float): The standard deviation.
            is_ligand (bool, optional): Whether the object represents a ligand. Defaults to True.
            angles (list[float], optional): List of angles. Defaults to None.
            procrustes_dists (list[float], optional): List of procrustes distances. Defaults to None.
        """
        self._ligand1 = ligand1
        self._ligand2 = ligand2
        self._angle = angle_value
        self._std = std if std > 1e-03 else 5
        self._is_ligand = is_ligand
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
    def is_ligand(self):
        """
        Check if the object represents a ligand.

        Returns:
            bool: True if the object represents a ligand, False otherwise.
        """
        return self._is_ligand

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

    def __init__(self, clazz, procrustes, coordination, count, description) -> None:
        """
        Initializes a LigandStats object.

        Args:
            clazz (str): The class of the ligand.
            procrustes (float): The procrustes value of the ligand.
            coordination (int): The coordination number of the ligand.
            count (int): The count of the ligand.
            description (str): The description of the ligand.
        """
        self._clazz = clazz
        self._procrustes = procrustes
        self._coordination = coordination
        self._count = count
        self._bonds = []
        self._pdb_bonds = []
        self._angles = []
        self._description = description
        self._cod_files = {}

    @property
    def clazz(self):
        """
        str: The class of the ligand.
        """
        return self._clazz

    @property
    def procrustes(self):
        """
        float: The procrustes value of the ligand.
        """
        return self._procrustes

    @property
    def coordination(self):
        """
        int: The coordination number of the ligand.
        """
        return self._coordination

    @property
    def count(self):
        """
        int: The count of the ligand.
        """
        return self._count

    @property
    def description(self):
        """
        str: The description of the ligand.
        """
        return self._description

    @property
    def cod_files(self):
        """
        dict: The names of COD files associated with the ligand.
        """
        return self._cod_files.keys()

    @property
    def cods(self):
            """
            Generator function that yields the key-value pairs of the _cod_files dictionary.

            Yields:
                tuple: A tuple containing the key and value of each item in the _cod_files dictionary.
            """
            for key, value in self._cod_files.items():
                yield key, value
    
    @property
    def bond_count(self):
        """
        int: The total number of bonds associated with the ligand.
        """
        return len(self._bonds) + len(self._pdb_bonds)

    @property
    def bonds(self):
        """
        Generator: Yields each bond associated with the ligand.
        """
        for bond in self._bonds:
            yield bond

    @property
    def pdb(self):
        """
        Generator: Yields each PDB bond associated with the ligand.
        """
        for bond in self._pdb_bonds:
            yield bond

    @property
    def angles(self):
        """
        Generator: Yields each angle associated with the ligand.
        """
        for ligand_angle in self._angles:
            yield ligand_angle

    @property
    def ligand_angles(self):
        """
        Generator: Yields each angle associated with the ligand that involves only ligands.
        """
        for ligand_angle in self._angles:
            if ligand_angle.is_ligand:
                yield ligand_angle

    def get_ligand_bond(self, ligand_name):
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

    def get_ligand_angle(self, ligand1_name, ligand2_name):
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

    def get_angle(self, ligand1_code, ligand2_code):
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

    def add_bond(self, distance):
        """
        Adds a bond to the ligand.

        Args:
            distance: The bond distance to add.
        """
        self._bonds.append(distance)

    def add_pdb_bond(self, distance):
        """
        Adds a PDB bond to the ligand.

        Args:
            distance: The PDB bond distance to add.
        """
        self._pdb_bonds.append(distance)

    def add_angle(self, new_angle):
        """
        Adds an angle to the ligand.

        Args:
            new_angle: The angle to add.
        """
        self._angles.append(new_angle)

    def add_cod_file(self, cod_file, structure):
        """
        Adds a COD file to the ligand.

        Args:
            cod_file: The COD file to add.
        """
        self._cod_files[cod_file] = structure

    def to_dict(self):
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


class MetalStats():
    def __init__(self, metal, metal_element, chain, residue, sequence, altloc, icode, mean_occ, mean_b) -> None:
        """
        Initialize a MetalStats object.

        Args:
            metal (str): The metal identifier.
            metal_element (str): The metal element.
            chain (str): The chain identifier.
            residue (str): The residue identifier.
            sequence (int): The sequence number.
            mean_occ (float): The mean occupancy.
            mean_b (float): The mean B-factor.
            altloc (str): The alternative location identifier.
            icode (str): The insertion code.
        """
        self._metal = metal
        self._metal_element = metal_element
        self._chain = chain
        self._residue = residue
        self._sequence = sequence
        self._mean_occ = mean_occ
        self._mean_b = mean_b
        self._icode = icode
        self._altloc = altloc
        self._ligands = []

    @property
    def code(self):
        """
        Get the code of the metal.

        Returns:
            tuple: A tuple containing the metal, metal element, chain, residue, and sequence.
        """
        return (self._metal, self._metal_element, self._chain, self._residue, str(self._sequence))

    @property
    def metal(self):
        """
        Get the metal identifier.

        Returns:
            str: The metal identifier.
        """
        return self._metal

    @property
    def metal_element(self):
        """
        Get the metal element.

        Returns:
            str: The metal element.
        """
        return self._metal_element

    @property
    def chain(self):
        """
        Get the chain identifier.

        Returns:
            str: The chain identifier.
        """
        return self._chain

    @property
    def residue(self):
        """
        Get the residue identifier.

        Returns:
            str: The residue identifier.
        """
        return self._residue

    @property
    def sequence(self):
        """
        Get the sequence number.

        Returns:
            int: The sequence number.
        """
        return self._sequence

    @property
    def altloc(self):
        """
        Get the alternative location identifier.

        Returns:
            str: The alternative location identifier.
        """
        return self._altloc
    
    @property
    def insertion_code(self):
        """
        Returns the insertion code of the metal.

        Returns:
            str: The insertion code of the metal.
        """
        if self._icode == "\u0000":
            return "."
        return self._icode if self._icode else "."

    @property
    def mean_occ(self):
        """
        Get the mean occupancy.

        Returns:
            float: The mean occupancy.
        """
        return self._mean_occ

    @property
    def mean_b(self):
        """
        Get the mean B-factor.

        Returns:
            float: The mean B-factor.
        """
        return self._mean_b

    @property
    def ligands(self):
        """
        Get an iterator over the ligands.

        Yields:
            Ligand: A ligand object.
        """
        for ligand in self._ligands:
            yield ligand

    def same_metal(self, other):
        """
        Check if two MetalStats objects represent the same metal.

        Args:
            other (MetalStats): Another MetalStats object.

        Returns:
            bool: True if the metals are the same, False otherwise.
        """
        return (self.metal == other.metal) and (self.metal_element == other.metalElement)

    def same_monomer(self, other):
        """
        Check if two MetalStats objects belong to the same monomer.

        Args:
            other (MetalStats): Another MetalStats object.

        Returns:
            bool: True if the monomers are the same, False otherwise.
        """
        return (self.chain == other.chain) and (self.residue == other.residue) and (self.sequence == other.sequence)

    def add_ligand(self, ligand):
        """
        Add a ligand to the MetalStats object.

        Args:
            ligand (Ligand): A ligand object.
        """
        self._ligands.append(ligand)

    def is_ligand_atom(self, atom):
        """
        Check if an atom is a ligand atom.

        Args:
            atom (Atom): An atom object.

        Returns:
            bool: True if the atom is a ligand atom, False otherwise.
        """
        return (atom.chain == self.chain) and (atom.residue == self.residue) and (atom.sequence == self.sequence)

    def get_coordination(self):
        """
        Get the maximum coordination number among the ligands.

        Returns:
            int: The maximum coordination number.
        """
        return np.max([l.coordination for l in self.ligands])

    def get_best_class(self):
        """
        Get the best class of ligands based on the coordination number and procrustes score.

        Returns:
            Ligand: The best class of ligands.
        """
        coord = self.get_coordination()
        return self._ligands[np.argmin([l.procrustes for l in self.ligands if l.coordination == coord])]

    def get_all_distances(self):
        """
        Get a list of all distances from the best class of ligands.

        Returns:
            list: A list of distances.
        """
        clazz = self.get_best_class()
        return list(clazz.bonds) + list(clazz.pdb)

    def get_ligand_distances(self):
        """
        Get a list of distances from the best class of ligands.

        Returns:
            list: A list of distances.
        """
        clazz = self.get_best_class()
        return list(clazz.bonds)

    def get_all_angles(self):
        """
        Get a list of all angles from the best class of ligands.

        Returns:
            list: A list of angles.
        """
        clazz = self.get_best_class()
        return list(clazz.angles)

    def get_ligand_angles(self):
        """
        Get a list of ligand angles from the best class of ligands.

        Returns:
            list: A list of ligand angles.
        """
        clazz = self.get_best_class()
        return [angle for angle in clazz.angles if self.is_ligand_atom(angle.ligand1) and self.is_ligand_atom(angle.ligand2)]

    def get_ligand_bond(self, ligand_name):
        """
        Get the bond associated with a ligand.

        Args:
            ligand_name (str): The name of the ligand.

        Returns:
            Bond: The bond associated with the ligand.
        """
        clazz = self.get_best_class()
        return clazz.get_ligand_bond(ligand_name)

    def get_ligand_angle(self, ligand1_name, ligand2_name):
        """
        Get the angle between two ligands.

        Args:
            ligand1_name (str): The name of the first ligand.
            ligand2_name (str): The name of the second ligand.

        Returns:
            Angle: The angle between the ligands.
        """
        clazz = self.get_best_class()
        return clazz.get_ligand_angle(ligand1_name, ligand2_name)

    def get_angle(self, ligand1_code, ligand2_code):
        """
        Get the angle between two ligands based on their codes.

        Args:
            ligand1_code (str): The code of the first ligand.
            ligand2_code (str): The code of the second ligand.

        Returns:
            Angle: The angle between the ligands.
        """
        clazz = self.get_best_class()
        return clazz.get_angle(ligand1_code, ligand2_code)

    def is_empty(self):
        """
        Check if the MetalStats object has any ligands.

        Returns:
            bool: True if the object has no ligands, False otherwise.
        """
        return len(self._ligands) == 0

    def to_dict(self):
        """
        Convert the MetalStats object to a dictionary.

        Returns:
            dict: A dictionary representation of the MetalStats object.
        """
        metal = {"chain": self.chain, "residue": self.residue, "sequence": self.sequence, "metal": self.metal,
                 "metalElement": self.metal_element, "icode": self.insertion_code, "altloc": self.altloc, "ligands": []}

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

    def metal_names(self):
        return self._metals.keys()

    def get_metal(self, metal_name):
        return self._metals.get(metal_name, None)

    def is_in(self, atom):
        return self._chain == atom.chain and self._residue == atom.residue and self._sequence == atom.sequence

    def add_metal(self, metal):
        if self.is_in(metal):
            self._metals.setdefault(metal.metal, metal)

    def contains(self, metal_name):
        return metal_name in self._metals

    def get_best_class(self, metal_name):
        if metal_name in self._metals:
            return self._metals[metal_name].get_best_class()
        return None

    def get_ligand_bond(self, metal_name, ligand_name):
        if metal_name in self._metals:
            return self._metals[metal_name].get_ligand_bond(ligand_name)
        return None

    def get_ligand_angle(self, metal_name, ligand1_name, ligand2_name):
        if metal_name in self._metals:
            return self._metals[metal_name].get_ligand_angle(ligand1_name, ligand2_name)
        return None

    def get_angle(self, metal_name, ligand1_code, ligand2_code):
        if metal_name in self._metals:
            return self._metals[metal_name].get_angle(ligand1_code, ligand2_code)
        return None

    def len(self):
        return len(self._metals)

    def is_empty(self):
        return self.len() == 0


class PdbStats():
    def __init__(self) -> None:
        self._monomers = dict()

    def add_metal(self, metal):
        if not metal.is_empty():
            self._monomers.setdefault(metal.chain + metal.residue + str(metal.sequence),
                                      MonomerStats(metal.chain, metal.residue, metal.sequence)).add_metal(metal)

    def monomers(self):
        for monomer in self._monomers.values():
            yield monomer

    def metal_names(self):
        return np.unique([name for monomer in self._monomers.values() for name in monomer.metal_names()]).tolist()

    @property
    def metals(self):
        for monomer in self._monomers.values():
            for metal in monomer.metals:
                yield metal

    def get_best_class(self, metal_name):

        metals = [monomer.get_metal(
            metal_name) for monomer in self.monomers() if monomer.contains(metal_name)]
        classes = [metal.get_best_class() for metal in metals]
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

    def get_ligand_distances(self, metal_name):
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_distances()
        return []

    def get_ligand_distance(self, metal_name, ligand_name):
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_bond(ligand_name)
        return None

    def get_ligand_angle(self, metal_name, ligand1_name, ligand2_name):
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_ligand_angle(ligand1_name, ligand2_name)
        return []

    def get_ligand_angles(self, metal_name):
        clazz = self.get_best_class(metal_name)

        if clazz:
            return clazz.ligand_angles
        return []

    def get_all_distances(self, metal_name):
        clazz = self.get_best_class(metal_name)
        if clazz:
            return clazz.get_all_distances()
        return []

    def is_empty(self):
        return len(self._monomers) == 0

    def len(self):
        return np.sum([monomer.len() for monomer in self.monomers()])

    def json(self):
        return [metal.to_dict() for monomer in self.monomers() for metal in monomer.metals]