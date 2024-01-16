from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from metalCoord.config import Config
from metalCoord.analysis.classes import idealClasses
from metalCoord.correspondense.procrustes import fit
from metalCoord.logging import Logger



def elementCode(code):
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return "".join(dict.fromkeys(result))


def elements(code):
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return result


class StatsData():

    def load(self):
        d = os.path.dirname(sys.modules["metalCoord"].__file__)
        self.__data = pd.read_csv(os.path.join(d, "data/classes.zip"))
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


def get_coordinate(file_data):
    return np.vstack([file_data[["MetalX", "MetalY", "MetalZ"]].values[:1], file_data[["LigandX", "LigandY", "LigandZ"]].values])


def get_groups(atoms1, atoms2):
    unique_atoms = np.unique(atoms1)
    group1 = []
    group2 = []
    for atom in unique_atoms:
        group1.append(np.where(atoms1 == atom)[0].tolist())
        group2.append(np.where(atoms2 == atom)[0].tolist())

    return [group1, group2]


def angle(metal, ligand1, ligand2):
    a = metal - ligand1
    b = metal - ligand2
    a = np.array(a)/np.linalg.norm(a)
    b = np.array(b)/np.linalg.norm(b)
    cosine_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine_angle))


class Ligand():
    def __init__(self, ligand) -> None:
        self._name = ligand.atom.name
        self._element = ligand.atom.element.name
        self._chain = ligand.chain.name
        self._residue = ligand.residue.name
        self._sequence = ligand.residue.seqid.num
    

    @property
    def name(self):
        return self._name
    
    @property
    def element(self):
        return self._element
    
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
    def code(self):
        return (self._name, self._element, self._chain, self._residue, self._sequence)
    
    def equals(self, other):
        return self.code == other.code
    
    def to_dict(self):
        return {"name": self.name, "element": self.element, "chain": self.chain, "residue": self.residue, "sequence ": self.sequence}

class DistanceStats():
    def __init__(self, ligand, distance, std) -> None:
        self._ligand = ligand
        self._distance = distance
        self._std = std if std > 1e-02 else 0.05
    
    @property
    def ligand(self):
        return self._ligand
    
    @property
    def distance(self):
        return self._distance
    
    @property
    def std(self):
        return self._std
    


class AngleStats():
    def __init__(self, ligand1, ligand2, angle, std, isLigand = True) -> None:
        self._ligand1 = ligand1
        self._ligand2 = ligand2
        self._angle = angle
        self._std =  std if std > 1e-03 else 5
        self._isLigand = isLigand
    
    @property
    def ligand1(self):
        return self._ligand1
    
    @property
    def ligand2(self):
        return self._ligand2
    
    @property
    def angle(self):
        return self._angle
    

    @property
    def std(self):
        return self._std
    
    @property
    def isLigand(self):
        return self._isLigand

    def equals(self, code1, code2):
        return (self.ligand1.code == code1 and self.ligand2.code == code2) or (self.ligand1.code == code2 and self.ligand2.code == code1)

class LigandStats():
    def __init__(self, clazz, procrustes, coordination, count) -> None:
        self._clazz = clazz
        self._procrustes = procrustes
        self._coordination = coordination
        self._count = count
        self._bonds = []
        self._pdb_bonds = []
        self._angles = []
    
    @property
    def clazz(self):
        return self._clazz
    
    @property
    def procrustes(self):
        return self._procrustes
    
    @property
    def coordination(self):
        return self._coordination
    
    @property
    def count(self):
        return self._count
    

    @property
    def bondCount(self):
        return len(self._bonds) + len(self._pdb_bonds)

    @property
    def bonds(self):
        for bond in self._bonds:
            yield bond
    
    @property
    def pdb(self):
        for bond in self._pdb_bonds:
            yield bond
    
    @property
    def angles(self):
        for angle in self._angles:
            yield angle
    
    @property
    def ligandAngles(self):
        for angle in self._angles:
            if angle.isLigand:
                yield angle
            


    def getLigandBond(self, ligand_name):
        for bond in self.bonds:
            if bond.ligand.name  == ligand_name:
                return bond
        return None

    def getLigandAngle(self, ligand1_name, ligand2_name):
        for angle in self.ligandAngles:
            if (angle.ligand1.name  == ligand1_name and angle.ligand2.name  == ligand2_name) or (angle.ligand1.name  == ligand2_name and angle.ligand2.name  == ligand1_name):
                return angle
        return None

    def getAngle(self, ligand1_code, ligand2_code):
        for angle in self.angles:
            if angle.equals(ligand1_code, ligand2_code):
                return angle
        return None
    
    def addBond(self, distance):
        self._bonds.append(distance)

    def addPdbBond(self, distance):
        self._pdb_bonds.append(distance)

    def addAngle(self, angle):
        self._angles.append(angle)

    def to_dict(self):
            clazz = {"class": self.clazz, "procrustes": np.round(float(self.procrustes), 3), "coordination": self.coordination,  "count": self.count, 
                     "base": [], "angles": [], "pdb": []}
            for b in self.bonds:
                clazz["base"].append(
                    {"ligand": b.ligand.to_dict(), "distance": b.distance, "std": b.std})
            for a in self.angles:
                clazz["angles"].append({"ligand1": a.ligand1.to_dict(),
                                        "ligand2": a.ligand2.to_dict(),
                                        "angle": a.angle, "std": a.std})
            for p in self.pdb:
                clazz["pdb"].append({"ligand": p.ligand.to_dict(), "distance": p.distance,
                                    "std": p.std })
            return clazz
        

class MetalStats():
    def __init__(self, metal, metalElement, chain, residue, sequence, description) -> None:
        self._metal = metal
        self._metalElement = metalElement
        self._chain = chain
        self._residue = residue
        self._sequence = sequence
        self._ligands = []
        self._description = description


    @property
    def code(self):
        return (self._metal, self._metalElement, self._chain, self._residue, str(self._sequence))
    
    @property
    def metal(self):
        return self._metal
    
    @property
    def metalElement(self):
        return self._metalElement
    
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
    def ligands(self):
        for ligand in self._ligands:
            yield ligand

    @property
    def description(self):
        return self._description
    
    def sameMetal(self, other):
        return (self.metal == other.metal) and (self.metalElement == other.metalElement)
    

    def sameMonomer(self, other):
         return (self.chain == other.chain) and (self.residue == other.residue) and (self.sequence == other.sequence)
    

    def addLigand(self, ligand):
        self._ligands.append(ligand)

    def isLigandAtom(self, atom):
        return (atom.chain == self.chain) and (atom.residue == self.residue) and (atom.sequence == self.sequence)

    def getCoordination(self):
        return np.max([l.coordination for l in self.ligands])
    
    def getBestClass(self):
        coord = self.getCoordination()
        return self._ligands[np.argmin([l.procrustes for l in self.ligands if l.coordination == coord])]

    def getAllDistances(self):
        clazz = self.getBestClass()
        return list(clazz.bonds) + list(clazz.pdb)
    
    def getLigandDistances(self):
        clazz = self.getBestClass()
        return list(clazz.bonds)
    
    def getAllAngles(self):
        clazz = self.getBestClass()
        return list(clazz.angles)
    
    def getLigandAngles(self):
        clazz = self.getBestClass()
        return [angle for angle in clazz.angles if self.isLigandAtom(angle.ligand1) and self.isLigandAtom(angle.ligand2)]

    def getLigandBond(self, ligand_name):
        clazz = self.getBestClass()
        return clazz.getLigandBond(ligand_name)
    
    def getLigandAngle(self, ligand1_name, ligand2_name):
        clazz = self.getBestClass()
        return clazz.getLigandAngle(ligand1_name, ligand2_name)
    
    def getAngle(self, ligand1_code, ligand2_code):
        clazz = self.getBestClass()
        return clazz.getAngle(ligand1_code, ligand2_code)
    
    def isEmpty(self):
        return len(self._ligands) == 0
    

    def to_dict(self):
        metal = {"chain": self.chain, "residue": self.residue, "sequence": self.sequence, "metal": self.metal,
                 "metalElement": self.metalElement, "ligands": [], "description": self.description}
        

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

    def metalNames(self):
        return self._metals.keys()

    def isIn(self, atom):
        return self._chain == atom.chain and self._residue == atom.residue and self._sequence == atom.sequence
    
    def addMetal(self, metal):
        if self.isIn(metal):
            self._metals.setdefault(metal.metal, metal)
    
    def contains(self, metal_name):
        return metal_name in self._metals

    def getBestClass(self, metal_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getBestClass()
        return None
    
    def getLigandBond(self, metal_name, ligand_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getLigandBond(ligand_name)
        return None
           
    
    def getLigandAngle(self, metal_name, ligand1_name, ligand2_name):
        if metal_name in self._metals:
            return self._metals[metal_name].getLigandAngle(ligand1_name, ligand2_name)
        return None
    
    def getAngle(self, metal_name, ligand1_code, ligand2_code):
        if metal_name in self._metals:
            return self._metals[metal_name].getAngle(ligand1_code, ligand2_code)
        return None

    def len(self):
        return len(self._metals)
     
    def isEmpty(self):
        return self.len() == 0
    
    

class PdbStats():
    def __init__(self) -> None:
        self._monomers = dict()
    
    def addMetal(self, metal):
        if not metal.isEmpty():
            self._monomers.setdefault(metal.chain + metal.residue + str(metal.sequence), MonomerStats(metal.chain, metal.residue, metal.sequence)).addMetal(metal)
    
    def monomers(self):
        for monomer in self._monomers.values():
            yield monomer

    def metalNames(self):
        return np.unique([name for monomer in self._monomers.values() for name in monomer.metalNames()]).tolist()
    
    @property
    def metals(self):
        for monomer in self._monomers.values():
            for metal in monomer.metals:
                yield metal
               
    def getBestClass(self, metal_name):
        classes = [monomer.getBestClass(metal_name) for monomer in self.monomers() if monomer.contains(metal_name)]
        coordinations = [clazz.coordination for clazz in classes]
        procrustes = [clazz.procrustes for clazz in classes]

        if np.min(coordinations) != np.max(coordinations):
            raise Exception(f"Different coordination numbers for the same metal {metal_name}. Please check pdb file.")
        
        return classes[np.argmin(procrustes)]

    def getLigandDistances(self, metal_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandDistances()
        return []
    
    def getLigandDistance(self, metal_name, ligand_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandBond(ligand_name)
        return None
    

    def getLigandAngle(self, metal_name, ligand1_name, ligand2_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getLigandAngle(ligand1_name, ligand2_name)
        return []
    
    def getLigandAngles(self, metal_name):
        clazz = self.getBestClass(metal_name)

        if clazz:
            return clazz.ligandAngles
        return []

    def getAllDistances(self, metal_name):
        clazz = self.getBestClass(metal_name)
        if clazz:
            return clazz.getAllDistances()
        return []
        
    def len(self):
        return len(self._metals)
     
    def isEmpty(self):
        return len(self._monomers) == 0
    
    def len(self):
        return np.sum([monomer.len() for monomer in self.monomers()])
    
    
    def json(self):
        return [metal.to_dict() for monomer in self.monomers() for metal in monomer.metals]
    

class CandidateFinder(ABC):

    def __init__(self) -> None:
        self._description = ""
        self._classes = None
        self._files =  None
        self._selection = None

    def load(self, structure, data):
        self._structure = structure
        self._data = data
        self._load()

    @abstractmethod
    def _load(self):
        pass


    def classes(self):
        return self._classes

    def files(self):
        return self._files

    def data(self, file):
        return self._selection[self._selection.File == file] if self._selection is not None else None

    def description(self):
        return self._description


class StrictCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Strict correspondence"
  
    def _load(self):
        self._selection = self._data[self._data.Code ==
                                      self._structure.code()]
        self._classes = self._selection.Class.unique()
        self._files = [self._selection[self._selection.Class ==
                                         cl].File.unique() for cl in self._classes]

class ElementCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on coordination, atom availability and atom count"

    def _load(self):
        code = elementCode(self._structure.code())
        self._selection = self._data[(self._data.ElementCode == code) & (
            self._data.Coordination == self._structure.coordination())]
        self._classes = self._selection.Class.unique()
        # Logger().info(f"Classes: {self._classes} Code: {code} Metal {self._structure.metal.name}")
        self._files = [self._selection[self._selection.Class ==
                                         cl].File.unique() for cl in self._classes]

class ElementInCandidateFinder(CandidateFinder):
    
    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination and all atoms availability only"

    def _load(self):
        code = elements(self._structure.code())
        coordinationData = self._data[(
            self._data.Coordination == self._structure.coordination())]
        self._selection = coordinationData[np.all(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]
        self._classes = self._selection.Class.unique()
        self._files = [self._selection[self._selection.Class ==
                                         cl].File.unique() for cl in self._classes]

class AnyElementCandidateFinder(CandidateFinder):
    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination and at least one atom availability"

    def _load(self):
        code = elements(self._structure.code())[1:]
        coordinationData = self._data[(self._data.Coordination == self._structure.coordination()) & (
            self._data.Metal == self._structure.metal.element.name)]
        self._selection = coordinationData[np.any(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]
        self._classes = self._selection.Class.unique()
        self._files = [self._selection[self._selection.Class ==
                                         cl].File.unique() for cl in self._classes]

class NoCoordinationCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on atom availability only"

    def _load(self):
        self._selection = self._data[(
            self._data.Metal == self._structure.metal.element.name)]
        self._classes = self._selection.Class.unique()
        self._files = [self._selection[self._selection.Class ==
                                         cl].File.unique() for cl in self._classes]


class StatsFinder(ABC):
    def __init__(self, candidateFinder) -> None:
        self._finder = candidateFinder
        self._thr = 0.3

    @abstractmethod
    def get_stats(self, structure, data):
        pass

    def _createStats(self, structure):
        return MetalStats(structure.metal.name, structure.metal.element.name, structure.chain.name, structure.residue.name, structure.residue.seqid.num, self._finder.description())


class FileStatsFinder(StatsFinder):
    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data):
        self._prepare(structure, data)
        return self._calculate(structure)

    def _prepare(self, structure, data):
        self._finder.load(structure, data)
        self._classes = self._finder.classes()
        self._files = self._finder.files()

    @abstractmethod
    def _calculate(self, stucture):
        pass


class StrictCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure):
        metalStats = self._createStats(structure)
        o_ligand_atoms = np.array([structure.metal.name] + structure.atoms())
        o_ligand_coord = structure.get_coord()

         
        for cl, files in zip(self._classes, self._files):
            if not idealClasses.contains(cl):
                continue
            m_ligand_coord = idealClasses.getCoordinates(cl)
            n_ligands = structure.coordination()
            main_proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)
            
            if main_proc_dist > self._thr:
                continue
            o_ligand_coord = m_ligand_coord[index]
        
            distances = []
            angles = []
            if len(files) > 2000:
                files = np.random.choice(files, 2000, replace=False)

            for file in tqdm(files, desc=f"{cl} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                m_ligand_atoms = np.insert(
                    file_data[["Ligand"]].values.ravel(), 0, structure.metal.name)
                groups = get_groups(o_ligand_atoms,  m_ligand_atoms)
                proc_dists, indices, min_proc_dist = fit(
                    o_ligand_coord, m_ligand_coord, groups=groups, all=True)
              
                for proc_dist, index in zip(proc_dists, indices):
                    if proc_dist >= Config().procrustes_thr():
                        continue

                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis=1))[1:].tolist())
                    angles.append([angle(m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j]) for i in range(
                        1, len(o_ligand_coord) - 1) for j in range(i + 1, len(o_ligand_coord))])

            distances = np.array(distances).T
            angles = np.array(angles).T
            if (distances.shape[0] > 0):
                clazzStats = LigandStats(
                    cl, main_proc_dist, structure.coordination(), distances.shape[1])
                n_ligands = len(structure.ligands)
                ligands = structure.ligands
                for i, l in enumerate(ligands):
                    dist, std = distances[i].mean(), distances[i].std()
                    clazzStats.addBond(DistanceStats(Ligand(l), dist, std))



                for i, l in enumerate(structure.extra_ligands):
                    dist, std = distances[i +
                                            n_ligands].mean(), distances[i + n_ligands].std()
                    clazzStats.addPdbBond(DistanceStats(Ligand(l), dist, std,))
                
                k = 0
                n_ligands = structure.coordination()
                n1 = len(ligands)
                ligands = ligands + structure.extra_ligands
                for i in range(n_ligands - 1):
                    for j in range(i + 1, n_ligands):
                        a, std = angles[k].mean(), angles[k].std()
                        clazzStats.addAngle(AngleStats(Ligand(ligands[i]), Ligand(ligands[j]), a, std, isLigand = i < n1 and j < n1))
                        k += 1


                metalStats.addLigand(clazzStats)
        return metalStats


class WeekCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure):
        metalStats = self._createStats(structure)
        o_ligand_coord = structure.get_coord()
        
        for cl, files in zip(self._classes, self._files):
            if not idealClasses.contains(cl):
                continue
            m_ligand_coord = idealClasses.getCoordinates(cl)
            n_ligands = structure.coordination()
            main_proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)
            
            if main_proc_dist > self._thr:
                continue
            o_ligand_coord = m_ligand_coord[index]
            

            distances = []
            ligNames = []
            if len(files) > 2000:
                files = np.random.choice(files, 2000, replace=False)
            for file in tqdm(files, desc=f"{cl} ligands", leave=False, disable=Logger().disabled):
                file_data = self._finder.data(file)
     
                m_ligand_coord = get_coordinate(file_data)
                proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)

                if proc_dist <  Config().procrustes_thr():
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[0] - m_ligand_coord)**2, axis=1))[1:].tolist())
                    ligNames.append(file_data[["Ligand"]].values.ravel().tolist())

            distances = np.array(distances).T
            ligNames = np.array(ligNames).T

            if (distances.shape[0] > 0):

                clazzStats = LigandStats(
                    cl, main_proc_dist, structure.coordination(), distances.shape[1])
                ligands = structure.ligands

                results = {}
                for element in np.unique(ligNames):
                    elementDistances = distances.ravel()[np.argwhere(
                        ligNames.ravel() == element)]
                    results[element] = (
                        elementDistances.mean(), elementDistances.std())
                    
               
                for i, l in enumerate(ligands):
                    dist, std = results[l.atom.element.name]
                    clazzStats.addBond(DistanceStats(Ligand(l), dist, std))

                for i, l in enumerate(structure.extra_ligands):
                    if l.atom.element.name in results:
                        dist, std = results[l.atom.element.name]
                        clazzStats.addPdbBond(DistanceStats(Ligand(l), dist, std))

                if idealClasses.contains(cl):

                    ligands = structure.ligands + structure.extra_ligands
                    m_ligand_coord = idealClasses.getCoordinates(cl)
                    n_ligands = structure.coordination()
                    proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)
                    n1 = len(structure.ligands)
                    for i in range(1, n_ligands):
                        for j in range(i + 1, n_ligands + 1):
                            a = angle( m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j])
                            std = 5.000
                            clazzStats.addAngle(AngleStats(Ligand(ligands[i - 1]), Ligand(ligands[j - 1]), a, std, isLigand = i <= n1 and j <= n1))

                metalStats.addLigand(clazzStats)
        return metalStats


class OnlyDistanceStatsFinder(StatsFinder):
    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data):
        self._finder.load(structure, data)
        data = self._finder.data("")
        metalStats = self._createStats(structure)
        clazzStats = LigandStats("", -1, structure.coordination(), -1)
        for l in structure.ligands:
            dist, std, count = DB.getDistanceStats(
                structure.metal.element.name, l.atom.element.name)
            if count > 0:
                clazzStats.addBond(DistanceStats(Ligand(l), dist, std))

        if len(structure.extra_ligands) > 0:
            for l in structure.extra_ligands:
                dist, std, count = DB.getDistanceStats(
                    structure.metal.element.name, l.atom.element.name)
                if count > 0:
                    clazzStats.addPdbBond(DistanceStats(Ligand(l), dist, std))
        if clazzStats.bondCount > 0:
            metalStats.addLigand(clazzStats)
        return metalStats
