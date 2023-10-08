
from metalCoord.correspondense.procrustes import fit
from metalCoord.analysis.classes import idealClasses
from abc import ABC, abstractmethod
import os
import sys
import pandas as pd
import numpy as np


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
        return self.__distances[(self.__distances.Metal == metal) & (self.__distances.Ligand == ligand)][["mean", "std", "count"]].values[0]

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
        self.name = ligand.atom.name
        self.element = ligand.atom.element.name
        self.chain = ligand.chain.name
        self.residue = ligand.residue.name
        self.sequence = ligand.residue.seqid.num


class DistanceStats():
    def __init__(self, ligand, distance, std) -> None:
        self.ligand = ligand
        self.distance = distance
        self.std = std


class AngleStats():
    def __init__(self, ligand1, ligand2, angle, std) -> None:
        self.ligand1 = ligand1
        self.ligand2 = ligand2
        self.angle = angle
        self.std = std


class LigandStats():
    def __init__(self, clazz, procrustes, coorditation, count) -> None:
        self.clazz = clazz
        self.procrustes = procrustes
        self.coorditation = coorditation
        self.count = count
        self.base = []
        self.pdb = []
        self.angles = []


class MetalStats():
    def __init__(self, metal, metalElement, chain, residue, sequence, description) -> None:
        self.metal = metal
        self.metalElement = metalElement
        self.chain = chain
        self.residue = residue
        self.sequence = sequence
        self.ligands = []
        self.description = description

    def addLigand(self, ligand):
        self.ligands.append(ligand)

    def isEmpty(self):
        return len(self.ligands) == 0


class CandidateFinder(ABC):

    def __init__(self) -> None:
        self._description = ""

    def load(self, structure, data):
        self._structure = structure
        self._data = data

    @abstractmethod
    def classes(self):
        pass

    @abstractmethod
    def files(self):
        pass

    @abstractmethod
    def data(self, file):
        pass

    def description(self):
        return self._description


class StrictCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Strict correspondence"

    def load(self, structure, data):
        super().load(structure, data)
        self.__selection = self._data[self._data.Code ==
                                      self._structure.code()]
        self.__classes = self.__selection.Class.unique()
        self.__files = [self.__selection[self.__selection.Class ==
                                         cl].File.unique() for cl in self.__classes]

    def classes(self):
        return self.__classes

    def files(self):
        return self.__files

    def data(self, file):
        return self.__selection[self.__selection.File == file]


class ElementCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination, atom availability and count"

    def load(self, structure, data):
        super().load(structure, data)
        code = elementCode(self._structure.code())
        self.__selection = self._data[(self._data.ElementCode == code) & (
            self._data.Coordination == self._structure.coordination())]
        self.__classes = self.__selection.Class.unique()
        self.__files = [self.__selection[self.__selection.Class ==
                                         cl].File.unique() for cl in self.__classes]

    def classes(self):
        return self.__classes

    def files(self):
        return self.__files

    def data(self, file):
        return self.__selection[self.__selection.File == file]


class ElementInCandidateFinder(CandidateFinder):
    def load(self, structure, data):
        super().load(structure, data)
        code = elements(self._structure.code())
        coordinationData = self._data[(
            self._data.Coordination == self._structure.coordination())]
        self.__selection = coordinationData[np.all(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]
        self.__classes = self.__selection.Class.unique()
        self.__files = [self.__selection[self.__selection.Class ==
                                         cl].File.unique() for cl in self.__classes]

    def classes(self):
        return self.__classes

    def files(self):
        return self.__files

    def data(self, file):
        return self.__selection[self.__selection.File == file]


class AnyElementCandidateFinder(CandidateFinder):
    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on cooordination and atom availability only"

    def load(self, structure, data):
        super().load(structure, data)
        code = elements(self._structure.code())[1:]
        coordinationData = self._data[(self._data.Coordination == self._structure.coordination()) & (
            self._data.Metal == self._structure.metal.element.name)]
        self.__selection = coordinationData[np.any(
            [coordinationData.ElementCode.str.contains(x) for x in code], axis=0)]
        self.__classes = self.__selection.Class.unique()
        self.__files = [self.__selection[self.__selection.Class ==
                                         cl].File.unique() for cl in self.__classes]

    def classes(self):
        return self.__classes

    def files(self):
        return self.__files

    def data(self, file):
        return self.__selection[self.__selection.File == file]


class NoCoordinationCandidateFinder(CandidateFinder):

    def __init__(self) -> None:
        super().__init__()
        self._description = "Based on atom availability only"

    def load(self, structure, data):
        super().load(structure, data)
        self.__selection = self._data[(
            self._data.Metal == self._structure.metal.element.name)]
        self.__classes = self.__selection.Class.unique()
        self.__files = [self.__selection[self.__selection.Class ==
                                         cl].File.unique() for cl in self.__classes]

    def classes(self):
        return self.__classes

    def files(self):
        return self.__files

    def data(self, file):
        return self.__selection


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
            distances = []
            angles = []
            for file in files:
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                m_ligand_atoms = np.insert(
                    file_data[["Ligand"]].values.ravel(), 0, structure.metal.name)
                groups = get_groups(o_ligand_atoms,  m_ligand_atoms)
                proc_dists, indices, min_proc_dist = fit(
                    o_ligand_coord, m_ligand_coord, groups=groups, all=True)
                if min_proc_dist < self._thr:
                    for _, index in zip(proc_dists, indices):
                        distances.append(np.sqrt(np.sum(
                            (m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis=1))[1:].tolist())
                        angles.append([angle(m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j]) for i in range(
                            1, len(o_ligand_coord) - 1) for j in range(i + 1, len(o_ligand_coord))])

            distances = np.array(distances).T
            angles = np.array(angles).T
            if (distances.shape[0] > 0):
                clazzStats = LigandStats(
                    cl, min_proc_dist, structure.coordination(), distances.shape[1])
                n_ligands = len(structure.ligands)
                ligands = structure.ligands
                for i, l in enumerate(ligands):
                    dist, std = distances[i].mean(), distances[i].std()
                    clazzStats.base.append(DistanceStats(Ligand(l), dist, std))



                for i, l in enumerate(structure.extra_ligands):
                    dist, std = distances[i +
                                            n_ligands].mean(), distances[i + n_ligands].std()
                    clazzStats.pdb.append(DistanceStats(Ligand(l), dist, std,))
                
                k = 0
                n_ligands = structure.coordination()
                ligands = ligands + structure.extra_ligands
                for i in range(n_ligands - 1):
                    for j in range(i + 1, n_ligands):
                        a, std = angles[k].mean(), angles[k].std()
                        clazzStats.angles.append(AngleStats(Ligand(ligands[i]), Ligand(ligands[j]), a, std))
                        k += 1


                metalStats.addLigand(clazzStats)
        return metalStats


class WeekCorrespondenceStatsFinder(FileStatsFinder):
    def _calculate(self, structure):
        metalStats = self._createStats(structure)
        o_ligand_coord = structure.get_coord()

        for cl, files in zip(self._classes, self._files):
            distances = []
            for file in files:
                file_data = self._finder.data(file)
                m_ligand_coord = get_coordinate(file_data)
                proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)

                if proc_dist < self._thr:
                    distances.append(np.sqrt(np.sum(
                        (m_ligand_coord[index][0] - m_ligand_coord[index])**2, axis=1))[1:].tolist())

            distances = np.array(distances).T

            if (distances.shape[0] > 0):

                clazzStats = LigandStats(
                    cl, proc_dist, structure.coordination(), distances.shape[1])
                ligands = structure.ligands

                results = {}
                atoms = np.array(structure.atoms())
                for element in np.unique(atoms):
                    elementDistances = distances[np.argwhere(
                        atoms == element)].ravel()
                    results[element] = (
                        elementDistances.mean(), elementDistances.std())

                for i, l in enumerate(ligands):
                    dist, std = results[l.atom.element.name]
                    clazzStats.base.append(DistanceStats(Ligand(l), dist, std))

                for i, l in enumerate(structure.extra_ligands):
                    dist, std = results[l.atom.element.name]
                    clazzStats.pdb.append(DistanceStats(Ligand(l), dist, std))

                if idealClasses.contains(cl):
                    ligands = structure.ligands + structure.extra_ligands
                    m_ligand_coord = idealClasses.getCoordinates(cl)
                    n_ligands = structure.coordination()
                    proc_dist, _, _, _, index = fit(o_ligand_coord, m_ligand_coord)
                    for i in range(1, n_ligands):
                        for j in range(i + 1, n_ligands + 1):
                            a = angle( m_ligand_coord[index][0], m_ligand_coord[index][i], m_ligand_coord[index][j])
                            std = 5.000
                            clazzStats.angles.append(AngleStats(Ligand(ligands[i - 1]), Ligand(ligands[j - 1]), a, std))

                metalStats.addLigand(clazzStats)
        return metalStats


class OnlyDistanceStatsFinder(StatsFinder):
    def __init__(self, candidateFinder) -> None:
        super().__init__(candidateFinder)

    def get_stats(self, structure, data):
        self._finder.load(structure, data)
        data = self._finder.data("")
        metalStats = self._createStats(structure)
        clazzStats = LigandStats("", -1, 0, -1)
        for l in structure.ligands:
            dist, std, count = DB.getDistanceStats(
                structure.metal.element.name, l.element.name)
            if count > 0:
                clazzStats.base.append(DistanceStats(Ligand(l), dist, std))

        if len(structure.extra_ligands) > 0:
            for l in structure.extra_ligands:
                dist, std, count = DB.getDistanceStats(
                    structure.metal.element.name, l.atom.element.name)
                if count > 0:
                    clazzStats.pdb.append(DistanceStats(Ligand(l), dist, std))
        if len(clazzStats.base) > 0:
            metalStats.addLigand(clazzStats)
        return metalStats
