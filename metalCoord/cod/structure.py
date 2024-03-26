import numpy as np
from abc import ABC
import gemmi
import os
from pathlib import Path
from tqdm import tqdm


class PDBReader:

    @staticmethod
    def getCoord(line):
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    @staticmethod
    def getElement(line):
        return line[76:78].strip()
    @staticmethod
    def getName(line):
        return line[12:16].strip()
    
    @staticmethod
    def isMeta(line):
        return (line[0:4] != 'HETA') and (line[0:4] != 'ATOM')


class Atom:
    def __init__(self, element, name, coords):
        self.__name = name
        self.__coords = coords
        self.__element = element
    
    @property
    def name(self):
        return self.__name

    @property
    def coords(self):
        return self.__coords
    @property
    def element(self):
        return self.__element
    @property
    def covrad(self):
        return gemmi.Element(self.__element).covalent_r
    @property
    def isMetal(self):
        return gemmi.Element(self.__element).is_metal
    @property
    def isH(self):
        return self.__element == 'H'
    
    @staticmethod
    def read(line):
        return Atom(PDBReader.getElement(line), PDBReader.getName(line), PDBReader.getCoord(line))


class Structure(ABC):

    def __init__(self, path = None):
        
        self._atoms = []
        self._coords = None
        self._path= path
    
    def atoms(self):
        for atom in self._atoms:
            yield atom

    def atom(self, name):
        for atom in self._atoms:
            if atom.name() == name:
                return atom
        return None
    
    @property
    def coords(self):
        if self._coords is None or len(self._coords) == len(self._atoms):
            self._coords = np.vstack([atom.coords for atom in self._atoms])
        return self._coords
    
    @property
    def centeredCoords(self):
        return self.coords - self.coords[0]
    
    @property
    def fileName(self):
        if self._path:
            return os.path.basename(self._path)
        return None
    
    @property
    def path(self):
        return self._path
    

    

    

    
class MetalStructure(Structure):
    def __init__(self, path = None):
        super().__init__(path=path)

        self._nonmetals = []
    @property
    def nonmetals(self):
        for nonmetal in self._nonmetals:
            yield nonmetal
            
    @property
    def names(self):
        for atom in self._atoms:
            yield atom.name()
    
    def addAtom(self, atom):
        if atom is None:
            return
        self._atoms.append(atom)
        self._nonmetals.append(atom)
    
    def addAtoms(self, atoms):
        if atoms is None:
            return
        self._atoms.extend(atoms)
        self._nonmetals.extend(atoms)

    
    

class SingleMetalStructure(MetalStructure):


    @staticmethod
    def read(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            count = 0
            structure = None
            for line in lines:

                if PDBReader.isMeta(line):
                    continue

                if count == 0:
                    metal = Atom.read(line)
                    structure = SingleMetalStructure(metal, path=path)
                    count = count + 1
                    continue

                ligand = Atom.read(line)
                if ligand.isH or ligand.isMetal:
                    continue
                structure.addAtom(ligand)
                count = count + 1

        return structure


    def __init__(self, metal, atoms = None, path = None):
        super().__init__(path=path)
        if metal is None:
           raise ValueError("No metal provided")
        
        self._metal = metal
              
        self._atoms.append(metal)
        self.addAtoms(atoms)
        

    @property
    def metal(self):
        return self._metal
    
    @property
    def distances(self):
        return np.linalg.norm(self._metal.coords - self.coords[1:], axis = 1)
    
    @property
    def coordination(self):
        return len(self._nonmetals)

    def cleanByCovRad(self, coeff = 1.3):
        distances = self.distances
        atoms = []
        for i,atom in enumerate(self._nonmetals):
            if distances[i] <= coeff * (atom.covrad + self._metal.covrad):
                atoms.append(atom)
        return SingleMetalStructure(self._metal, atoms)
    
    
class Pattern(SingleMetalStructure):
    def __init__(self, structure):
        super().__init__(structure.metal, path = structure.path)
        self._class = self.fileName.replace(".pdb", "").lower()
        for s in structure.nonmetals:
            self.addAtom(s)

    @property
    def cl(self):
        return self._class
    
class MultipleMetalStructure(MetalStructure):
    def __init__(self, metals, atoms = None):
        super().__init__()
        if not metals:
           raise ValueError("No metals provided")
        self._metals = []
        self._metals.extend(metals)
        self.addAtoms(atoms)

    @property
    def metals(self):
        for metal in self._metals:
            yield metal
    
    def metal(self, name):
        for metal in self._metals:
            if metal.name() == name:
                return metal
        return None
    
    def distances(self, name):
        metal = self.metal(name)
        if metal:
            return np.linalg.norm(metal.coords- self.coords[len(self._metals):], axis = 1)
        return None
    
    def allDistances(self):
        for metal in self._metals:
            yield np.linalg.norm(metal.coords - self.coords[len(self._metals):], axis = 1)


class PatternDB:
    @staticmethod
    def read(folder):
        rootdir = Path(folder)
        db = PatternDB()
        files = [str(f) for f in rootdir.glob('**/*pdb*') if f.is_file()]
        for file in tqdm(files, desc="Patterns loading", unit="p"):
            db.add(Pattern(SingleMetalStructure.read(file)))
        return db

    def __init__(self):
        self._patterns = dict()
    
    def add(self, pattern):
        coordinationPatterns = self._patterns.get(pattern.coordination, [])
        coordinationPatterns.append(pattern)
        self._patterns[pattern.coordination] = coordinationPatterns
    
    def classNames(self, coordination):
        return [pattern.cl for pattern in self._patterns.get(coordination, [])]
    
    def classes(self, coordination):
        return [pattern for pattern in self._patterns.get(coordination, [])]
    
    def queryInCoordination(self, coordination, func, *args):
        return [pattern for pattern in self._patterns.get(coordination, []) if func(pattern.cl, *args)]
    
    def queryNamesInCoordination(self, coordination, func, *args):
        return [pattern.cl for pattern in self.queryInCoordination(coordination, func, *args)]
    
    def query(self, func, *args):
        return [pattern for pattern in self.all if func(pattern.cl, *args)]
    
    def queryNames(self, func, *args):
        return [pattern.cl for pattern in self.query(func, *args)]
    
    @property
    def all(self):
        for coordination in self._patterns:
            for pattern in self._patterns[coordination]:
                yield pattern
    @property
    def allNames(self):
        for coordination in self._patterns:
            for pattern in self._patterns[coordination]:
                yield pattern.cl
    @property
    def coordinations(self):
        return sorted(self._patterns.keys())
    

def clean_sm_distance(input_path, output_path, coeff=1):
    with open(input_path, 'r') as file:
        lines = file.readlines()
        count = 0

        out_lines = []
        for line in lines:
            if((line[0:4] != 'HETA') and (line[0:4] != 'ATOM')):
                out_lines.append(line)
                continue
            
            
            if count == 0:
                out_lines.append(line)
                count = count + 1
                metal = line[76:78].strip()
                metal_coord = coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                continue
            
            ligand = line[76:78].strip()
            
            if ligand == 'H':
                continue

            if gemmi.Element(ligand).is_metal:
                    continue
            
            
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        
            if np.sqrt(np.sum((coord - metal_coord)**2)) > coeff*(gemmi.Element(ligand).covalent_r + gemmi.Element(metal).covalent_r):
                continue
            
            out_lines.append(line)
            count = count + 1

        if count < 3:
            file.close()
            return None

        with open(output_path, 'w') as out_file:
            for line in out_lines:
                out_file.write(line)

