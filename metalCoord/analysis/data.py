import os
import sys
from typing import Optional

import pandas as pd

from metalCoord.analysis.utlis import elementCode


class StatsData:
    """
    A class that represents statistical data for metal coordination analysis.

    Attributes:
        __data (pandas.DataFrame): The loaded data for analysis.
        __distances (pandas.DataFrame): The calculated distances between metals and ligands.

    Methods:
        load(): Loads the data for analysis.
        getDistanceStats(metal, ligand): Retrieves the distance statistics for a specific metal and ligand.
        data(): Returns the loaded data for analysis.
    """

    def __init__(self, data_path: Optional[str] = None):
        """Initializes the StatsData object."""

        self.__data = None
        self.__distances = None
        self.__data_path = data_path

    def _data_path(self) -> str:
        if self.__data_path:
            return self.__data_path
        module_path = os.path.dirname(sys.modules["metalCoord"].__file__)
        return os.path.join(module_path, "data/classes.zip")

    def clear_cache(self) -> None:
        """Clear cached datasets to force a reload on next access."""

        self.__data = None
        self.__distances = None

    def load(self) -> None:
        """Loads the data for analysis if it has not been loaded yet."""

        if self.__data is not None and self.__distances is not None:
            return
        self.__data = pd.read_csv(self._data_path(), keep_default_na=False)
        self.__data.loc[self.__data.index, "Code"] = self.__data.File.map(
            self.__data.groupby("File").Ligand.agg(lambda x: "".join(sorted(x)))
        )
        self.__data.loc[self.__data.index, "Code"] = (
            self.__data.Metal + self.__data.Code
        )
        self.__data = self.__data[self.__data.Metal != self.__data.Metal.str.lower()]
        self.__data["ElementCode"] = self.__data.Code.apply(elementCode)
        self.__data["COD"] = self.__data["File"].str[:7]
        self.__distances = (
            self.__data.groupby(["Metal", "Ligand"])
            .Distance.agg(["mean", "std", "count"])
            .reset_index()
        )

    def ensure_loaded(self) -> None:
        """Ensure that the cached dataset has been initialised."""

        self.load()

    def get_distance_stats(self, metal, ligand):
        """
        Retrieves the distance statistics for a specific metal and ligand.

        Args:
            metal (str): The metal element.
            ligand (str): The ligand element.

        Returns:
            tuple: A tuple containing the mean, standard deviation, and count of distances.
        """
        self.ensure_loaded()
        result = self.__distances[
            (self.__distances.Metal == metal) & (self.__distances.Ligand == ligand)
        ][["mean", "std", "count"]].values
        return result[0] if len(result) > 0 else (0, 0, 0)

    def get_frequency(self, cod=False):
        """
        Retrieves the frequency of each class for all metals and coordinations.

        cod (bool): Whether to include the COD IDs in the result.

        Returns:
            dict: The frequency of each class for all metals and coordinations.
        """
        return self._get_stats(self.__data, cod=cod)

    def get_frequency_coordination(self, coordination, cod=False):
        """
        Retrieves the frequency of each class for the specific coordination.

        Args:
            coordination (str): The coordination to search for.
            cod (bool): Whether to include the COD IDs in the result.

        Returns:
            dict: The frequency of each class for the coordination.
        """
        self.ensure_loaded()
        selection = self.__data.loc[self.__data.Coordination == coordination]
        data = self._get_stats(selection, cod=cod)
        sorted_data = sorted(
            data.items(), key=lambda x: x[1]["frequency"], reverse=True
        )
        return {key: value for key, value in sorted_data}

    def get_frequency_metal_coordination(self, metal, coordination, cod=False):
        """
        Retrieves the frequency of each class for the specific coordination for a given metal.

        Args:
            metal (str): The metal element.
            coordination (str): The coordination to search for.

        Returns:
            int: The frequency of each class for the coordination for the given metal.
        """

        self.ensure_loaded()
        selection = self.__data.loc[
            (self.__data.Metal == metal) & (self.__data.Coordination == coordination)
        ]
        data = self._get_stats(selection, cod=cod)
        sorted_data = sorted(
            data.items(), key=lambda x: x[1]["frequency"], reverse=True
        )
        return {key: value for key, value in sorted_data}

    def get_frequency_metal(self, metal, cod=False):
        """
        Retrieves the frequency of each class for all coordinations for a given metal.

        Args:
            metal (str): The metal element.
            cod (bool): Whether to include the COD IDs in the result.
        Returns:
            dict: The frequency of each class for all coordinations for the given metal.
        """

        self.ensure_loaded()
        selection = self.__data.loc[self.__data.Metal == metal]
        return self._get_stats(selection, cod=cod)

    def _get_stats(self, data, cod=False):
        """
        Calculate statistics for the given data.

        Parameters:
        data (pandas.DataFrame): The input data containing at least 'Class', 'File', and 'Coordination' columns.
        cod (bool, optional): If True, include 'COD' column in the aggregation. Defaults to False.

        Returns:
        dict: A dictionary where keys are the unique values from the 'Class' column and values are dictionaries
              containing 'frequency' (proportion of each class), 'coordination' (first value of 'Coordination' column),
              and optionally 'cod' (sorted list of unique 'COD' values if cod is True).
        """
        self.ensure_loaded()
        group = data.groupby("Class")
        agg = {"File": "count", "Coordination": "first"}
        if cod:
            agg["COD"] = "unique"
        stats = group.agg(agg)
        stats["Freq"] = stats["File"] / data.shape[0]
        result = dict()
        for index, row in stats.iterrows():
            result[index] = {
                "frequency": float(row["Freq"]),
                "coordination": int(row["Coordination"]),
                "count": int(row["File"]),
            }
            if cod:
                result[index]["cod"] = sorted(row["COD"].tolist())
        return dict(sorted(result.items(), key=lambda item: item[1]["coordination"]))

    def data(self):
        """
        Returns the loaded data for analysis.

        Returns:
            pandas.DataFrame: The loaded data for analysis.
        """
        self.ensure_loaded()
        return self.__data


DB = StatsData()
