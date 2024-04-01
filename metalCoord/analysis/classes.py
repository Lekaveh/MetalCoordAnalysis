import os
import sys
import pandas as pd



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
        

idealClasses = Class()