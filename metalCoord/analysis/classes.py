import operator
import os
import sys
import pandas as pd
from metalCoord.correspondense.procrustes import fit


class Class():
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

    def get_ideal_classes(self: str):
        """
        Retrieves the names of the ideal classes.

        Returns:
            dict_keys: The names of the ideal classes.
        """
        return [key for key, _ in sorted(self.__classes.items(), key = operator.itemgetter(1))]

    def get_ideal_classes_by_coordination(self, coordination_num: int):
        """
        Retrieves the names of the ideal classes with a specific coordination.

        Args:
            coordination_num (int): The coordination number.

        Returns:
            dict_keys: The names of the ideal classes with the specified coordination.
        """
        return [clazz for clazz in self.__classes.keys() if self.__classes[clazz] == coordination_num + 1]
    
    def get_coordination(self, class_name: str):
        """
        Retrieves the coordination number of a class.

        Args:
            class_name (str): The name of the class.

        Returns:
            int: The coordination number of the class.
        """
        return self.__classes.get(class_name, 1) - 1


class ClassificationResult():
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


class Classificator():
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

    def classify(self, structure):
        """
        Classifies a structure.

        Args:
            structure: The structure to classify.

        Yields:
            ClassificationResult: The classification results.
        """
        for clazz in idealClasses.get_ideal_classes():
            if idealClasses.get_coordination(clazz) != structure.coordination():
                continue
            m_ligand_coord = idealClasses.get_coordinates(clazz)
            main_proc_dist, _, _, _, index = fit(
                structure.get_coord(), m_ligand_coord)
            if main_proc_dist < self._thr:
                yield ClassificationResult(clazz, m_ligand_coord, index, main_proc_dist)


idealClasses = Class()
