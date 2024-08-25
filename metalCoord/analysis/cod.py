from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from metalCoord.analysis.utlis import elementCode, elements

class CandidateFinder(ABC):
    """
    Abstract base class for finding candidates.
    
    Attributes:
        _description (str): Description of the candidate finder.
        _classes (np.ndarray): Classes of candidates.
        _files (dict): Files of candidates.
        _selection (pd.DataFrame): Selected candidates.
    """

    def __init__(self) -> None:
        """
        Initialize the CandidateFinder with default attributes.
        """
        self._description = ""
        self._classes = None
        self._files = None
        self._selection = None

    def load(self, structure, data: pd.DataFrame) -> None:
        """
        Load the structure and data into the candidate finder.
        
        Args:
            structure: The structure to load.
            data (pd.DataFrame): The data to load.
        """
        self._structure = structure
        self._data = data
        self._load()
        self._classes = self._selection.Class.unique()
        self._files = {cl: self._selection[self._selection.Class == cl].File.unique() for cl in self._classes}

    @abstractmethod
    def _load(self) -> None:
        """
        Abstract method to load data. Must be implemented by subclasses.
        """
        pass

    def classes(self) -> np.ndarray:
        """
        Get the classes of candidates.
        
        Returns:
            np.ndarray: Array of classes.
        """
        return self._classes

    def files(self) -> dict:
        """
        Get the files of candidates.
        
        Returns:
            dict: Dictionary of files.
        """
        return self._files

    def data(self, file: str) -> pd.DataFrame:
        """
        Get the data for a specific file.
        
        Args:
            file (str): The file to get data for.
        
        Returns:
            pd.DataFrame: Data for the file.
        """
        return self._selection[self._selection.File == file] if self._selection is not None else None

    def description(self) -> str:
        """
        Get the description of the candidate finder.
        
        Returns:
            str: Description of the candidate finder.
        """
        return self._description



class StrictCandidateFinder(CandidateFinder):
    """
    A candidate finder that uses strict correspondence to find candidates.

    This class inherits from the CandidateFinder base class and implements a candidate finder
    that uses strict correspondence to find candidates. It overrides the _load method to perform
    the candidate selection based on strict correspondence.

    Attributes:
        _description (str): A string representing the description of the candidate finder.

    Methods:
        __init__(): Initializes the StrictCandidateFinder object.
        _load(): Loads the data and performs candidate selection based on strict correspondence.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Strict correspondence"

    def _load(self):
        """
        Loads the data and performs candidate selection based on strict correspondence.

        This method overrides the _load method of the base class. It selects candidates based on
        strict correspondence, where the code of the candidate matches the code of the structure.

        Returns:
            None
        """
        self._selection = self._data[self._data.Code == self._structure.code()]


class ElementCandidateFinder(CandidateFinder):
    """
    A candidate finder based on coordination, atom availability, and atom count.

    This class is responsible for finding element candidates based on their coordination,
    atom availability, and atom count. It inherits from the `CandidateFinder` class.

    Attributes:
        _description (str): A description of the candidate finder.

    Methods:
        __init__(): Initializes the ElementCandidateFinder object.
        _load(): Loads the data and performs the candidate selection based on coordination and element code.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on coordination, atom availability and atom count"

    def _load(self):
        """
        Load data based on the element code and coordination of the structure.

        This method retrieves data from the `_data` attribute based on the element code
        and coordination of the structure. It assigns the filtered data to the `_selection`
        attribute.

        Parameters:
        None

        Returns:
        None
        """
        code = elementCode(self._structure.code())
        self._selection = self._data[(self._data.ElementCode == code) & (
            self._data.Coordination == self._structure.coordination())]


class ElementInCandidateFinder(CandidateFinder):
    """
    A candidate finder based on coordination and availability of all atoms.

    This class inherits from the CandidateFinder class and provides a specific implementation
    for finding candidates based on coordination and the availability of all atoms.

    Attributes:
        _description (str): A description of the candidate finder.

    Methods:
        _load(): Loads the coordination data and selects candidates based on coordination number and element codes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on coordination and all atoms availability only"

    def _load(self):
        """
        Load the coordination data based on the structure's code and coordination number.

        Returns:
            None
        """
        code = elements(self._structure.code())
        coordination_data = self._data[(
            self._data.Coordination == self._structure.coordination())]
        self._selection = coordination_data[np.all(
            [coordination_data.ElementCode.str.contains(x) for x in code], axis=0)]


class AnyElementCandidateFinder(CandidateFinder):
    """
    A candidate finder based on coordination and at least one atom availability.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on coordination and at least one atom availability"

    def _load(self):
        """
        Load coordination data based on the structure's code, coordination number, and metal.

        Returns:
            None
        """
        code = elements(self._structure.code())[1:]
        coordination_data = self._data[(self._data.Coordination == self._structure.coordination()) & (
            self._data.Metal == self._structure.metal.atom.element.name)]
        self._selection = coordination_data[np.any(
            [coordination_data.ElementCode.str.contains(x) for x in code], axis=0)]


class NoCoordinationCandidateFinder(CandidateFinder):
    """
    A candidate finder based on atom availability only.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on atom availability only"

    def _load(self):
        """
        Load the data into the selection based on the metal element name.

        This method retrieves the data from the `_data` attribute and filters it based on the
        metal element name of the current structure. The filtered data is then stored in the
        `_selection` attribute.

        """
        self._selection = self._data[(
            self._data.Metal == self._structure.metal.atom.element.name)]


class CovalentCandidateFinder(CandidateFinder):
    """
    A candidate finder based on covalent distances.
    """

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on covalent distances"

    def _load(self):
        """
        Load the data based on the metal element name.

        This method loads the data from the `_data` attribute based on the metal
        element name obtained from the `_structure` attribute. It assigns the
        filtered data to the `_selection` attribute.

        Parameters:
            None

        Returns:
            None
        """
        self._selection = self._data[self._data.Metal ==
                                     self._structure.metal.atom.element.name]