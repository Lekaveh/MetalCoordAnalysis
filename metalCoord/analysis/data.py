import os
import sys
import pandas as pd
from metalCoord.analysis.utlis import elementCode



class StatsData():
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

    def __init__(self):
        """
        Initializes the StatsData object.
        """
        self.__data = None
        self.__distances = None

    def load(self):
        """
        Loads the data for analysis.
        """
        d = os.path.dirname(sys.modules["metalCoord"].__file__)
        self.__data = pd.read_csv(os.path.join(d, "data/classes.zip"))
        self.__data.loc[self.__data.index, 'Code'] = self.__data.File.map(self.__data.groupby('File').Ligand.agg(lambda x: "".join(sorted(x))))
        self.__data.loc[self.__data.index, "Code"] = self.__data.Metal + self.__data.Code
        self.__data = self.__data[self.__data.Metal != self.__data.Metal.str.lower()]
        self.__data["ElementCode"] = self.__data.Code.apply(elementCode)
        self.__distances = self.__data.groupby(["Metal", "Ligand"]).Distance.agg([
            "mean", "std", "count"]).reset_index()

    def get_distance_stats(self, metal, ligand):
        """
        Retrieves the distance statistics for a specific metal and ligand.

        Args:
            metal (str): The metal element.
            ligand (str): The ligand element.

        Returns:
            tuple: A tuple containing the mean, standard deviation, and count of distances.
        """
        result = self.__distances[(self.__distances.Metal == metal) & (self.__distances.Ligand == ligand)][["mean", "std", "count"]].values
        return result[0] if len(result) > 0 else (0, 0, 0)
    
    def get_frequency(self, metal, coordination):
        """
        Retrieves the frequency of each class for the specific coordination for a given metal.

        Args:
            metal (str): The metal element.
            coordination (str): The coordination to search for.

        Returns:
            int: The frequency of each class for the coordination for the given metal.
        """

        selection = self.__data.loc[(self.__data.Metal == metal) & (self.__data.Coordination == coordination)]
        return (selection.groupby("Class")["File"].count()/selection.shape[0]).to_dict()

    def get_frequency_all(self, metal):
        """
        Retrieves the frequency of each class for all coordinations for a given metal.

        Args:
            metal (str): The metal element.

        Returns:
            dict: The frequency of each class for all coordinations for the given metal.
        """

        selection = self.__data.loc[self.__data.Metal == metal]
        return (selection.groupby(["Class"])["File"].count()/selection.shape[0]).to_dict()
        
    def data(self):
        """
        Returns the loaded data for analysis.

        Returns:
            pandas.DataFrame: The loaded data for analysis.
        """
        return self.__data


DB = StatsData()
DB.load()