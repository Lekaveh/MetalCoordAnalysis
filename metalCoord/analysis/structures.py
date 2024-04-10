import gemmi
import numpy as np
from metalCoord.config import Config


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
    
    def contains(self, atom):
        return atom in [ligand.atom for ligand in self.ligands + self.extra_ligands]
    
    def filter(self):
        to_delete = []
        for i, atom1 in enumerate(self.extra_ligands):
            for atom2 in self.ligands:
                if distance(atom1.atom, atom2.atom) < 1:
                    to_delete.append(i)
                    break
        if to_delete:
           self.extra_ligands = [atom for i, atom in enumerate(self.extra_ligands) if i not in to_delete]

    def mean_occ(self):
        return np.mean([self.metal.occ] + [ligand.atom.occ for ligand in self.ligands + self.extra_ligands])
    
    def mean_b(self):
        return np.mean([self.metal.b_iso] + [ligand.atom.b_iso for ligand in self.ligands + self.extra_ligands])

    def __str__(self) -> str:
        return f"{self.metal.name} - {self.chain.name} - {self.residue.name} - {self.residue.seqid.num}"
        
        

def distance(atom1, atom2):
    return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

def get_ligands(st, ligand, bonds = {}, max_dist = 10):
    scale = Config().scale()
    def covalent_radii(element):
        return gemmi.Element(element).covalent_r

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
                    metal_name = atom.name
                    metal_bonds = set(bonds.get(metal_name, []))
                    ligand_obj = Ligand(atom, residue, chain)
                    structures.append(ligand_obj)

                    marks = ns.find_neighbors(atom, min_dist=0.1, max_dist=max_dist)
                    for mark in marks:
                        cra = mark.to_cra(st[0])
                        if cra.atom.element.is_metal:
                            continue
                        if ligand_obj.contains(cra.atom):
                            continue
                        if bonds:
                            if cra.residue.name == ligand:
                                if cra.atom.name in metal_bonds:
                                    ligand_obj.ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                            elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                ligand_obj.extra_ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                        elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                            if cra.residue.name == ligand:
                                ligand_obj.ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                            else:
                                ligand_obj.extra_ligands.append(Atom(cra.atom, cra.residue, cra.chain))

                    ligand_obj.filter()

                    if len(ligand_obj.ligands) < len(bonds.get(metal_name, [])):
                        raise Exception(f"There is inconsistency between ligand(s) in the PDB and monomer file. Metal {metal_name} in {chain.name} - {residue.name} - {residue.seqid.num} has fewer neighbours than expected. Expected: {bonds.get(metal_name, [])}, found: {[l.atom.name for l in ligand_obj.ligands]}")

    return structures