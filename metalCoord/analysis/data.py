import os
import sys
import numpy as np
import pandas as pd
from metalCoord.analysis.utlis import elementCode



class StatsData():

    def load(self):
        d = os.path.dirname(sys.modules["metalCoord"].__file__)
        self.__data = pd.read_csv(os.path.join(d, "data/classes.zip"))
        self.__data.loc[self.__data.index, 'Code'] = self.__data.File.map(self.__data.groupby('File').Ligand.agg(lambda x: "".join(sorted(x))))
        self.__data.loc[self.__data.index, "Code"] = self.__data.Metal + self.__data.Code
        self.__data = self.__data[self.__data.Metal != self.__data.Metal.str.lower()]
        self.__data["ElementCode"] = self.__data.Code.apply(
            lambda x: elementCode(x))
        self.__distances = self.__data.groupby(["Metal", "Ligand"]).Distance.agg([
            "mean", "std", "count"]).reset_index()

    def getDistanceStats(self, metal, ligand):
        result = self.__distances[(self.__distances.Metal == metal) & (self.__distances.Ligand == ligand)][["mean", "std", "count"]].values
        return result[0] if len(result) > 0 else (0, 0, 0)

    def data(self):
        return self.__data


DB = StatsData()
DB.load()