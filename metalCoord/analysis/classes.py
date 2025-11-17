import operator
import os
import sys
import numpy as np
import pandas as pd
from metalCoord.analysis.data import DB
from metalCoord.analysis.structures import Ligand
from metalCoord.correspondense.procrustes import fit

MOST_COMMON_CLASS = "most_common"


# Inverse mapping: coordination number (or "<cn>_sandwich") -> {code: readable_name}
IDEAL_CLASS_MAP = {
    "linear": "LIN",
    "bent": "BEN",
    "trigonal-planar": "TPL",
    "t-shape": "TSH",
    "pyramid": "PYR",
    "capped-linear": "CLN",
    "tetrahedral": "TET",
    "square-planar": "SPL",
    "trigonal-pyramid": "TPY",
    "square-non-planar": "SNP",
    "bicapped-linear": "BLN",
    "capped-trigonal-planar": "CTP",
    "trigonal-bipyramid": "TBP",
    "square-pyramid": "SQP",
    "bicapped-trigonal-planar": "BTP",
    "tricapped-trigonal-planar": "TTP",
    "capped-square-planar": "CSP",
    "octahedral": "OCT",
    "trigonal-prism": "TRP",
    "bicapped-square-planar": "BSP",
    "hexagonal-planar": "HPL",
    "sandwich_4h_2": "SAA",
    "sandwich_4_2": "SAB",
    "sandwich_5_1": "SAC",
    "pentagonal-bipyramid": "PBP",
    "capped-trigonal-prism": "CTR",
    "capped-octahedral": "COP",
    "sandwich_6_1": "SAD",
    "sandwich_4_3": "SAE",
    "sandwich_4h_3": "SAF",
    "sandwich_5_2": "SAG",
    "square-antiprismatic": "SAP",
    "hexagonal-bipyramid": "HBP",
    "cubic": "CUB",
    "dodecahedral": "DOD",
    "bicapped-octahedral": "BOC",
    "elongated-triangular-bipyramid": "ETB",
    "sandwich_4h_4h": "SAH",
    "sandwich_7_1": "SAI",
    "sandwich_4h_4": "SAJ",
    "sandwich_5_capped_1": "SAK",
    "sandwich_5h_3": "SAL",
    "sandwich_6_2": "SAM",
    "sandwich_5_3": "SAN",
    "tricapped-trigonal-prismatic": "TCP",
    "sandwich_7_2": "SAO",
    "sandwich_5_tricapped_i": "SAP",
    "sandwich_5_4h": "SAQ",
    "sandwich_5_4": "SAR",
    "sandwich_6_3": "SAS",
    "pentagonal-antiprismatic": "PAP",
    "bicapped-square-antiprismatic": "BSA",
    "sandwich_5_5o": "SAT",
    "sandwich_6_trigonal_pyramid": "SAU",
    "sandwich_6_4": "SAV",
    "sandwich_5_4_i": "SAW",
    "sandwich_7_3": "SAX",
    "sandwich_5_tricapped_v": "SAY",
    "sandwich_5_square_pyramid": "SAZ",
    "sandwich_5_5": "SBA",
    "sandwich_5_pentagon_pyramid": "SBB",
    "sandwich_5_capped_square_pyramid": "SBC",
    "sandwich_8_3": "SBD",
    "sandwich_5_4h_v": "SBE",
    "sandwich_5_5_i": "SBF",
    "sandwich_6_5": "SBG",
    "sea_mine": "SEA",
    "cuboctahedron": "COB",
    "paired_octahedral": "POD",
    "sandwich_5_hexagon_pyramid": "SBH",
    "sandwich_8_4": "SBI",
    "sandwich_7_5": "SBJ",
    "sandwich_6_6": "SBK",
    "sandwich_5_5_v": "SBL",
    "sandwich_6_5_v": "SBM",
    "sandwich_8_5": "SBN",
    "sandwich_5_5_vi": "SBO",
    "sandwich_5_5_v_v_3d": "SBP",
    "sandwich_6_6_v": "SBQ",
    "sandwich_c5_5_i": "SBR",
    "sandwich_5_5_v_v": "SBS",
    "sandwich_8_5_i": "SBT",
    "sandwich_5_5_4": "SBU",
    "sandwich_6_6_triangle": "SBV",
    "sandwich_5_5_star": "SBW",
    "sandwich_8_8": "SBX",
    "sandwich_8_8_i": "SBY",
    "sandwich_8_8_v": "SBZ",
    "ball": "BAL",
}


class Class:
    """
    Represents a class of data points.

    Attributes:
        __data (pandas.DataFrame): The data containing the coordinates and classes.
        __classes (dict): A dictionary mapping class names to their counts.
    """

    def __init__(self) -> None:
        """
        Initializes a Class object.

        Reads the data from a CSV file and calculates the class counts.
        """
        d = os.path.dirname(sys.modules["metalCoord"].__file__)
        self.__data = pd.read_csv(os.path.join(d, "data/ideal.csv"))
        self.__classes = self.__data.groupby("Class").size().to_dict()

    def contains(self, class_name: str):
        """
        Checks if a class exists in the data.

        Args:
            class_name (str): The name of the class to check.

        Returns:
            bool: True if the class exists, False otherwise.
        """
        return class_name in self.__data["Class"].values

    def get_coordinates(self, class_name: str):
        """
        Retrieves the coordinates of a class.

        Args:
            class_name (str): The name of the class.

        Returns:
            numpy.ndarray: The coordinates of the class.
        """
        return self.__data[self.__data["Class"] == class_name][["X", "Y", "Z"]].values

    def get_ideal_classes(self):
        """
        Retrieves the names of the ideal classes.

        Returns:
            dict_keys: The names of the ideal classes.
        """
        return [
            key for key, _ in sorted(self.__classes.items(), key=operator.itemgetter(1))
        ]

    def get_ideal_classes_by_coordination(self, coordination_num: int):
        """
        Retrieves the names of the ideal classes with a specific coordination.

        Args:
            coordination_num (int): The coordination number.

        Returns:
            dict_keys: The names of the ideal classes with the specified coordination.
        """
        return [
            clazz
            for clazz in self.__classes.keys()
            if self.__classes[clazz] == coordination_num + 1
        ]

    def get_coordination(self, class_name: str):
        """
        Retrieves the coordination number of a class.

        Args:
            class_name (str): The name of the class.

        Returns:
            int: The coordination number of the class.
        """
        return self.__classes.get(class_name, 1) - 1

    def get_class_code(self, class_name: str):
        """
        Retrieves the code of a class.

        Args:
            class_name (str): The name of the class.

        Returns:
            str: The code of the class.
        """
        return IDEAL_CLASS_MAP.get(class_name, "UNK")


class ClassificationResult:
    """
    Represents the result of a classification.

    Attributes:
        _clazz (str): The class name.
        _coord (numpy.ndarray): The coordinates of the class.
        _index (int): The index of the class in the data.
        _proc (float): The Procrustes distance.
    """

    def __init__(self, clazz, coord, index, proc) -> None:
        """
        Initializes a ClassificationResult object.

        Args:
            clazz (str): The class name.
            coord (numpy.ndarray): The coordinates of the class.
            index (int): The index of the class in the data.
            proc (float): The Procrustes distance.
        """
        self._clazz = clazz
        self._coord = coord
        self._index = index
        self._proc = proc

    @property
    def clazz(self):
        """
        Gets the class name.

        Returns:
            str: The class name.
        """
        return self._clazz

    @property
    def coord(self):
        """
        Gets the coordinates of the class.

        Returns:
            numpy.ndarray: The coordinates of the class.
        """
        return self._coord

    @property
    def index(self):
        """
        Gets the index of the class in the data.

        Returns:
            int: The index of the class.
        """
        return self._index

    @property
    def order(self):
        """
        Gets the order of the class in the data.

        Returns:
            int: The order of the class.
        """
        return np.argsort(self._index)

    @property
    def proc(self):
        """
        Gets the Procrustes distance.

        Returns:
            float: The Procrustes distance.
        """
        return self._proc

    def __str__(self) -> str:
        """
        Returns a string representation of the ClassificationResult object.

        Returns:
            str: The string representation.
        """
        return f"Class: {self.clazz}, Procrustes: {self.proc}, Coordination: {len(self.coord) - 1}"


class Classificator:
    """
    Represents a classificator.

    Attributes:
        _thr (float): The threshold for the Procrustes distance.
    """

    def __init__(self, thr=0.3) -> None:
        """
        Initializes a Classificator object.

        Args:
            thr (float): The threshold for the Procrustes distance.
        """
        self._thr = thr

    @property
    def threshold(self):
        """
        Gets the threshold for the Procrustes distance.

        Returns:
            float: The threshold.
        """
        return self._thr

    def classify(self, structure: Ligand, class_name: str = None):
        """
        Classifies a structure.

        Args:
            structure: The structure to classify.
            class_name (str): The class name to classify the structure

        Yields:
            ClassificationResult: The classification results.
        """

        """Skipping structures with coordination number > 17 due to performance issues."""
        if structure.coordination() > 17:
            return

        if class_name:
            if class_name == MOST_COMMON_CLASS:
                classes = DB.get_frequency_metal_coordination(
                    structure.metal.element, structure.coordination()
                )
                if not classes:
                    return
                clazz = next(iter(classes.keys()), None)

            else:
                if not idealClasses.contains(class_name):
                    raise ValueError(f"Class {class_name} not found.")
                if structure.coordination() < idealClasses.get_coordination(class_name):
                    raise ValueError(
                        f"Class {class_name} has higher coordination number {idealClasses.get_coordination(class_name)} than {structure.name_code_with_symmetries()}."
                    )

                if (
                    structure.coordination()
                    > idealClasses.get_coordination(class_name) + 1
                ):
                    raise ValueError(
                        f"Class {class_name} has lower coordination number {idealClasses.get_coordination(class_name)} than {structure.name_code_with_symmetries()}."
                    )

                if (
                    structure.coordination()
                    == idealClasses.get_coordination(class_name) + 1
                ):
                    return
                clazz = class_name

            m_ligand_coord = idealClasses.get_coordinates(clazz)
            main_proc_dist, _, _, _, index = fit(structure.get_coord(), m_ligand_coord)
            yield ClassificationResult(clazz, m_ligand_coord, index, main_proc_dist)
        else:
            for clazz in idealClasses.get_ideal_classes():
                if idealClasses.get_coordination(clazz) != structure.coordination():
                    continue
                m_ligand_coord = idealClasses.get_coordinates(clazz)
                main_proc_dist, _, _, _, index = fit(
                    structure.get_coord(), m_ligand_coord
                )
                if main_proc_dist < self._thr:
                    yield ClassificationResult(
                        clazz, m_ligand_coord, index, main_proc_dist
                    )


idealClasses = Class()
