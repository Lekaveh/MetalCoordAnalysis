import numpy as np
from .chemistry import radiuses
import gemmi

class Atom:
    def __init__(self, atom, residue, chain):
        self.atom = atom
        self.residue = residue
        self.chain = chain

class Ligand:
    def __init__(self, metal, residue, chain) -> None:
        self.metal = metal
        self.residue = residue
        self.chain = chain
        self.ligands = []
        self.extra_ligands = []
    def elements(self):
        return sorted([ligand.atom.element.name for ligand in self.ligands + self.extra_ligands])
    def atoms(self):
        return [ligand.atom.element.name for ligand in self.ligands + self.extra_ligands]
    def code(self):
        return "".join([self.metal.element.name] + self.elements())
    def get_coord(self):
        return np.array([self.metal.pos.tolist()] + [ligand.atom.pos.tolist() for ligand in self.ligands + self.extra_ligands])
    def coordination(self):
        return len(self.ligands) + len(self.extra_ligands)
        

def distance(atom1, atom2):
    return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

def get_ligands(st, ligand, scale = 1.2, max_dist = 5):
    if st is None:
        return None
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate(include_h=False)
    structures = []
    for chain in st[0]:
        for residue in chain:
            if residue.name != ligand:
                continue
            for atom in residue:
                if atom.element.is_metal:
                    structures.append(Ligand(atom, residue, chain))
                    marks = ns.find_neighbors(atom, min_dist=0.1, max_dist=max_dist)
                    for mark in marks:
                        cra = mark.to_cra(st[0])
                        if cra.atom.element.is_metal:
                            continue
                        
                        if distance(atom, cra.atom) > (radiuses[atom.element.name] + radiuses[cra.atom.element.name])*scale:
                            continue
 
                        if cra.residue.name == ligand:
                            structures[-1].ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                        else:
                            structures[-1].extra_ligands.append(Atom(cra.atom, cra.residue, cra.chain))


    return structures