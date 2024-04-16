import gemmi
import numpy as np
from metalCoord.config import Config


class Atom:
    def __init__(self, atom, residue, chain):
        self._atom = atom
        self._residue = residue
        self._chain = chain
    
    @property
    def atom(self):
        return self._atom
    
    @property
    def residue(self):
        return self._residue
    
    @property
    def chain(self):
        return self._chain



class Ligand:
    def __init__(self, metal, residue, chain) -> None:
        self._metal = metal
        self._residue = residue
        self._chain = chain
        self._ligands = []
        self._extra_ligands = []



    def _euclidean(self, atom1:gemmi.Atom, atom2:gemmi.Atom):
        return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))
    
    def _dist(self, l:Atom):
        return self._euclidean(self.metal, l.atom)
    
    def _cov_dist(self, l:Atom):
        return gemmi.Element(self.metal.element.name).covalent_r + gemmi.Element(l.atom.element.name).covalent_r
    
    def _cov_dist_coeff(self, l:Atom):
        return self._euclidean(self.metal, l.atom) / self._cov_dist(l)
    
    def cov_dist_coeffs(self):
        return {l.atom.name:self._cov_dist_coeff(l) for l in self._extra_ligands}

    @property
    def metal(self):
        return self._metal
    @property
    def residue(self):
        return self._residue
    @property
    def chain(self):
        return self._chain
    
    @property
    def ligands(self):
        for ligand in self._ligands:
            yield ligand

    @property
    def extra_ligands(self):
        for ligand in self._extra_ligands:
            yield ligand

    @property
    def all_ligands(self):
        for ligand in self._ligands + self._extra_ligands:
            yield ligand
 
    @property
    def ligands_len(self):
        return len(self._ligands)
    
    def elements(self):
        return sorted([ligand.atom.element.name for ligand in self._ligands + self._extra_ligands])
    
    def atoms(self):
        return [ligand.atom.element.name for ligand in self._ligands + self._extra_ligands]
    
    
    def code(self):
        return "".join([self._metal.element.name] + self.elements())
    
    def get_coord(self):
        return np.array([self._metal.pos.tolist()] + [ligand.atom.pos.tolist() for ligand in self._ligands + self._extra_ligands])
    
    def coordination(self):
        return len(self._ligands) + len(self._extra_ligands)
    
    def contains(self, atom):
        return atom in [ligand.atom for ligand in self._ligands + self._extra_ligands]
    
    def filter_extra(self):
        to_delete = []
        for i, atom1 in enumerate(self._extra_ligands):
            for atom2 in self._ligands:
                if distance(atom1.atom, atom2.atom) < 1:
                    to_delete.append(i)
                    break
        if to_delete:
           self._extra_ligands = [atom for i, atom in enumerate(self._extra_ligands) if i not in to_delete]

        
    def filter_base(self):
        to_delete = []
        for i, atom1 in enumerate(self._ligands):
            if self._cov_dist_coeff(atom1) > 2:
                to_delete.append(i)

        if to_delete:
           self._ligands = [atom for i, atom in enumerate(self._ligands) if i not in to_delete]
        
        


    def mean_occ(self):
        return np.mean([self._metal.occ] + [ligand.atom.occ for ligand in self._ligands + self._extra_ligands])
    
    def mean_b(self):
        return np.mean([self._metal.b_iso] + [ligand.atom.b_iso for ligand in self._ligands + self._extra_ligands])

    def __str__(self) -> str:
        return f"{self._metal.name} - {self._chain.name} - {self._residue.name} - {self._residue.seqid.num}"
        
        

def distance(atom1, atom2):
    return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

def get_ligands(st, ligand, bonds = {}, max_dist = 10, only_best = False):
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
                            
                            if cra.residue.name == ligand and cra.residue.seqid.num  == residue.seqid.num and cra.chain.name == chain.name:
                            # if cra.residue.name == ligand:
                                if cra.atom.name in metal_bonds:
                                    ligand_obj._ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                            elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                ligand_obj._extra_ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                        elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                            if cra.residue.name == ligand and cra.residue.seqid.num  == residue.seqid.num and cra.chain.name == chain.name:
                                ligand_obj._ligands.append(Atom(cra.atom, cra.residue, cra.chain))
                            else:
                                ligand_obj._extra_ligands.append(Atom(cra.atom, cra.residue, cra.chain))

                    # ligand_obj.filter_base()
                    ligand_obj.filter_extra()

                    if len(ligand_obj._ligands) < len(bonds.get(metal_name, [])):
                        raise Exception(f"There is inconsistency between ligand(s) in the PDB and monomer file. Metal {metal_name} in {chain.name} - {residue.name} - {residue.seqid.num} has fewer neighbours than expected. Expected: {bonds.get(metal_name, [])}, found: {[l.atom.name for l in ligand_obj._ligands]}")


    if only_best:
        best_structures = []
        metals = np.unique([structure.metal.name for structure in structures]).tolist()
        
        for metal_name in metals:
            metal_stuctures = [s for s in structures if metal_name == s.metal.name]
            b_vlaues = [metal.mean_b() for metal in metal_stuctures]
            occ_values = [metal.mean_occ() for metal in metal_stuctures]
            best_occ = np.max(occ_values)
            best_structures.append(metal_stuctures[np.argmin(np.where(occ_values == best_occ, b_vlaues, np.inf))]) 
        structures = best_structures  
    
    return structures
