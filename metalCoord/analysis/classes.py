import os
import sys
import pandas as pd
from metalCoord.correspondense.procrustes import fit


class Class():
    def __init__(self) -> None:
        d = os.path.dirname(sys.modules["metalCoord"].__file__)
        self.__data = pd.read_csv(os.path.join(d, "data/ideal.csv"))
        self.__classes = self.__data.groupby("Class").size().to_dict()
    
    def contains(self, className):
        return className in self.__data["Class"].values
    
    def getCoordinates(self, className):
        return self.__data[self.__data["Class"] == className][["X", "Y", "Z"]].values
    
    def getIdealClasses(self):
        return self.__classes.keys()
    
    def getCoordination(self, className):
        return self.__classes.get(className, 1) - 1

class ClassificationResult():
    def __init__(self, clazz, coord, index, proc) -> None:
        self._clazz = clazz
        self._coord = coord
        self._index = index
        self._proc = proc

    @property
    def clazz(self):
        return self._clazz
    
    @property
    def coord(self):
        return self._coord
    
    @property
    def index(self):
        return self._index
    
    @property
    def proc(self):
        return self._proc
    
    def __str__(self) -> str:
        return f"Class: {self.clazz}, Procrustes: {self.proc}, Coordination: {len(self.coord) - 1}"    
    
class Classificator():
    def __init__(self, thr = 0.3) -> None:
        self._thr = thr

    def classify(self, structure):
        for clazz in idealClasses.getIdealClasses():
            if idealClasses.getCoordination(clazz) != structure.coordination():
                continue
            m_ligand_coord = idealClasses.getCoordinates(clazz)
            main_proc_dist, _, _, _, index = fit(structure.get_coord(), m_ligand_coord)
            if main_proc_dist < self._thr:
                yield ClassificationResult(clazz, m_ligand_coord, index, main_proc_dist)

idealClasses = Class()