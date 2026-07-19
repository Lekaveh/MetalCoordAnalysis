from collections import Counter
from typing import Iterable, Optional

import gemmi
import networkx as nx
import numpy as np

from metalCoord.analysis.structures import Atom, IAtom, Ligand

GROUPED_CANDIDATE_ATTR = "_metalcoord_grouped_view"

# Handle prefix for centroid pseudo-atoms (NMR/chemistry convention: Q1, Q2, ...)
CENTROID_NAME_PREFIX = "Q"


def _pos_array(atom: IAtom) -> np.ndarray:
    pos = atom.pos
    return np.array([pos.x, pos.y, pos.z], dtype=float)


def _position(values: np.ndarray) -> gemmi.Position:
    return gemmi.Position(float(values[0]), float(values[1]), float(values[2]))


def _clean_altloc(altloc: str) -> str:
    value = (altloc or "").strip().replace("\x00", "")
    return "" if value in (".", "?") else value


def _clean_icode(icode: str) -> str:
    return (icode or "").strip().replace("\x00", "")


def _atom_altloc(atom: IAtom) -> str:
    return _clean_altloc(atom.atom.altloc)


def _atom_key(atom: IAtom) -> tuple:
    return (
        atom.chain.name,
        atom.residue.name,
        atom.residue.seqid.num,
        _clean_icode(atom.residue.seqid.icode),
        atom.atom.name,
        _atom_altloc(atom),
        atom.symmetry,
        atom.symmetry_operator,
    )


def _cra_key(cra) -> tuple:
    return (
        cra.chain.name,
        cra.residue.name,
        cra.residue.seqid.num,
        _clean_icode(cra.residue.seqid.icode),
        cra.atom.name,
        _clean_altloc(cra.atom.altloc),
        0,
        "x,y,z",
    )


def _is_graph_atom(atom: gemmi.Atom) -> bool:
    return not atom.element.is_metal and atom.element.name not in ("H", "D")


def _same_altloc(atom1: IAtom, atom2: IAtom) -> bool:
    return _atom_altloc(atom1) == _atom_altloc(atom2)


def _can_group(atom1: IAtom, atom2: IAtom) -> bool:
    if atom1.symmetry or atom2.symmetry:
        return False
    return _same_altloc(atom1, atom2)


def _representative_element(atoms: list[IAtom]) -> str:
    counts = Counter(atom.atom.element.name for atom in atoms)
    max_count = max(counts.values())
    return sorted(element for element, count in counts.items() if count == max_count)[0]


def _atom_name_with_symmetry(atom: IAtom) -> str:
    return f"{atom.name} {'(' + str(atom.symmetry) + ')' if atom.symmetry else ''}"


def _atom_debug(atom: IAtom) -> dict:
    icode = _clean_icode(atom.residue.seqid.icode)
    return {
        "name": atom.atom.name,
        "element": atom.atom.element.name,
        "chain": atom.chain.name,
        "residue": atom.residue.name,
        "sequence": atom.residue.seqid.num,
        "icode": icode if icode else ".",
        "altloc": _atom_altloc(atom),
        "symmetry": atom.symmetry,
        "operator": atom.symmetry_operator,
    }


class CentroidAtom(IAtom):
    """
    Internal classification-only atom representing a collapsed ligand group.
    """

    def __init__(self, group_number: int, atoms: list[IAtom]) -> None:
        self._members = atoms
        element = _representative_element(atoms)
        pos = np.mean([_pos_array(atom) for atom in atoms], axis=0)

        self._atom = gemmi.Atom()
        self._atom.name = f"{CENTROID_NAME_PREFIX}{group_number}"
        self._atom.element = gemmi.Element(element)
        self._atom.pos = _position(pos)
        self._atom.occ = float(np.mean([atom.atom.occ for atom in atoms]))
        self._atom.b_iso = float(np.mean([atom.atom.b_iso for atom in atoms]))
        self._atom.altloc = "\x00"

        self._residue = atoms[0].residue
        self._chain = atoms[0].chain

    @property
    def atom(self):
        return self._atom

    @property
    def name(self):
        return self._atom.name

    @property
    def element(self):
        return self._atom.element.name

    @property
    def residue(self):
        return self._residue

    @property
    def chain(self):
        return self._chain

    @property
    def symmetry(self):
        return 0

    @property
    def translation(self):
        return (0, 0, 0)

    @property
    def symmetry_operator(self):
        return "x,y,z"

    @property
    def pos(self):
        return self._atom.pos

    @property
    def members(self):
        return self._members


class GroupedLigandView:
    """
    Classification-only view where bonded ligand groups are represented by centroids.
    """

    def __init__(
        self,
        structure: Ligand,
        groups: list[list[IAtom]],
        added: Optional[list[list[IAtom]]] = None,
    ) -> None:
        self._structure = structure
        self._groups = groups
        # Bridge ("added") atoms pulled in while forming each group. They are
        # marked here but never contribute to the centroid or to any restraint:
        # the centroid and grouped_base / grouped_angles use `groups` (the
        # original metal-bonded donors) only.
        self._added = added if added is not None else [[] for _ in groups]
        self._atoms = [
            CentroidAtom(i + 1, group) if len(group) > 1 else group[0]
            for i, group in enumerate(groups)
        ]

    @property
    def original_structure(self) -> Ligand:
        return self._structure

    @property
    def metal(self):
        return self._structure.metal

    @property
    def residue(self):
        return self._structure.residue

    @property
    def chain(self):
        return self._structure.chain

    @property
    def groups(self) -> list[list[IAtom]]:
        return self._groups

    @property
    def all_ligands(self) -> Iterable[IAtom]:
        for atom in self._atoms:
            yield atom

    @property
    def collapsed_group_count(self) -> int:
        return sum(1 for group in self._groups if len(group) > 1)

    @property
    def original_coordination(self) -> int:
        return self._structure.coordination()

    def group_members(self, view_index: int) -> list[IAtom]:
        return self._groups[view_index]

    def group_added_members(self, view_index: int) -> list[IAtom]:
        """Bridge atoms added while forming this group (excluded from output)."""
        return self._added[view_index]

    def atom_names_with_symmetries(self):
        return [self._structure.metal.name] + [
            _atom_name_with_symmetry(atom) for atom in self._atoms
        ]

    def element_names(self):
        return [self._structure.metal.atom.element.name] + [
            atom.atom.element.name for atom in self._atoms
        ]

    def descriptor_tokens(self):
        """Descriptor tokens with each collapsed group rendered as ``Q``.

        Mirrors ``element_names()`` but substitutes ``CENTROID_NAME_PREFIX`` for a
        group vertex, so the descriptor marks a grouped site without leaking the
        group's representative element. Ordering stays element-based via
        ``element_names()``; this only changes the displayed token.
        """
        return [self._structure.metal.atom.element.name] + [
            CENTROID_NAME_PREFIX
            if isinstance(atom, CentroidAtom)
            else atom.atom.element.name
            for atom in self._atoms
        ]

    def name_code_with_symmetries(self):
        return " ".join(self.atom_names_with_symmetries())

    def atoms(self):
        return [atom.atom.element.name for atom in self._atoms]

    def get_coord(self):
        return np.array(
            [self._structure.metal.pos.tolist()]
            + [atom.pos.tolist() for atom in self._atoms]
        )

    def coordination(self):
        return len(self._atoms)

    def debug_groups(self) -> list[dict]:
        groups = []
        for i, (view_atom, members, added) in enumerate(
            zip(self._atoms, self._groups, self._added), start=1
        ):
            groups.append(
                {
                    "view_index": i,
                    "centroid": {
                        "name": view_atom.atom.name,
                        "element": view_atom.atom.element.name,
                        "position": view_atom.pos.tolist(),
                    },
                    "members": [_atom_debug(atom) for atom in members],
                    "added_members": [_atom_debug(atom) for atom in added],
                }
            )
        return groups


def _graph_from_cif(structure: Ligand) -> tuple[nx.Graph, dict]:
    graph = nx.Graph()
    for atom_name, neighbors in structure.nonmetal_bonds.items():
        graph.add_node(atom_name)
        for neighbor in neighbors:
            graph.add_edge(atom_name, neighbor)
    # No coordinate atoms are available on the CIF path, so bridge nodes cannot be
    # materialised into IAtoms; the node index stays empty (best-effort).
    return graph, {}


def _graph_from_model(structure: Ligand) -> tuple[nx.Graph, dict]:
    st = getattr(structure.metal, "structure", None)
    if st is None:
        return nx.Graph(), {}

    graph = nx.Graph()
    model = st[0]
    ns = gemmi.NeighborSearch(model, st.cell, 4.2).populate(include_h=False)
    records = []
    # key -> (gemmi.Atom, gemmi.Residue, gemmi.Chain), so a bridge node can later
    # be materialised into an `Atom` on demand (only for the few bridge keys).
    node_index = {}
    for chain in model:
        for residue in chain:
            for atom in residue:
                if not _is_graph_atom(atom):
                    continue
                key = (
                    chain.name,
                    residue.name,
                    residue.seqid.num,
                    _clean_icode(residue.seqid.icode),
                    atom.name,
                    _clean_altloc(atom.altloc),
                    0,
                    "x,y,z",
                )
                graph.add_node(key)
                node_index[key] = (atom, residue, chain)
                records.append((key, atom))

    for key, atom in records:
        for mark in ns.find_neighbors(atom, min_dist=0.1, max_dist=4.2):
            if mark.image_idx != 0:
                continue
            cra = mark.to_cra(model)
            if not _is_graph_atom(cra.atom):
                continue
            other_key = _cra_key(cra)
            if other_key == key:
                continue
            if _clean_altloc(atom.altloc) != _clean_altloc(cra.atom.altloc):
                continue
            max_dist = 1.3 * (atom.element.covalent_r + cra.atom.element.covalent_r)
            if atom.pos.dist(cra.atom.pos) <= max_dist:
                graph.add_edge(key, other_key)

    return graph, node_index


def _ligand_graph_key(structure: Ligand, atom: IAtom):
    if structure.nonmetal_bonds:
        return atom.atom.name
    return _atom_key(atom)


def _materialise_bridges(bridge_keys, node_index, st) -> list[IAtom]:
    """Turn bridge node keys into `Atom` IAtoms (model path only).

    On the CIF path `node_index` is empty (no coordinate atoms), so bridges are
    recorded as none — they are excluded from every output surface regardless.
    """
    atoms = []
    if not node_index or st is None:
        return atoms
    for key in bridge_keys:
        meta = node_index.get(key)
        if meta is None:
            continue
        atom, residue, chain = meta
        atoms.append(Atom(atom, residue, chain, None, st))
    return atoms


def create_grouped_view(
    structure: Ligand, max_bridge_atoms: int = 2
) -> Optional[GroupedLigandView]:
    """
    Collapse coordinated non-metal atoms connected within the ligand graph.
    """
    ligands = list(structure.all_ligands)
    if len(ligands) < 2:
        return None

    if structure.nonmetal_bonds:
        graph, node_index = _graph_from_cif(structure)
    else:
        graph, node_index = _graph_from_model(structure)
    st = getattr(structure.metal, "structure", None)
    max_edges = max_bridge_atoms + 1

    # Donor atoms (the metal-bonded set that exists *before* the group search).
    # Any node lying on a connecting path that is itself a donor is not a bridge.
    donor_keys = {_ligand_graph_key(structure, atom) for atom in ligands}

    ligand_graph = nx.Graph()
    ligand_graph.add_nodes_from(range(len(ligands)))
    edge_bridge_keys = {}

    for i, atom1 in enumerate(ligands):
        for j in range(i + 1, len(ligands)):
            atom2 = ligands[j]
            if not _can_group(atom1, atom2):
                continue
            key1 = _ligand_graph_key(structure, atom1)
            key2 = _ligand_graph_key(structure, atom2)
            if key1 not in graph or key2 not in graph:
                continue
            try:
                path = nx.shortest_path(graph, key1, key2)
            except nx.NetworkXNoPath:
                continue
            if len(path) - 1 > max_edges:
                continue
            ligand_graph.add_edge(i, j)
            # Interior nodes of the connecting path that are NOT themselves donors
            # are the "added" bridge atoms pulled in while forming the group. A
            # donor (e.g. a carboxylate CD linking OE1/OE2) stays a full member --
            # it is bonded to the metal, so it belongs in the group and centroid.
            edge_bridge_keys[(i, j)] = [
                key for key in path[1:-1] if key not in donor_keys
            ]

    components = [
        sorted(component) for component in nx.connected_components(ligand_graph)
    ]
    components.sort(key=lambda component: component[0])

    groups = []
    added = []
    for component in components:
        component_set = set(component)
        groups.append([ligands[i] for i in component])
        bridge_keys = []
        seen = set()
        for (i, j), keys in edge_bridge_keys.items():
            if i in component_set and j in component_set:
                for key in keys:
                    if key not in seen:
                        seen.add(key)
                        bridge_keys.append(key)
        added.append(_materialise_bridges(bridge_keys, node_index, st))

    if not any(len(group) > 1 for group in groups):
        return None

    return GroupedLigandView(structure, groups, added)


def mark_grouped_candidate(class_result, grouped_view: GroupedLigandView):
    setattr(class_result, GROUPED_CANDIDATE_ATTR, grouped_view)
    return class_result


def is_grouped_candidate(class_result) -> bool:
    return getattr(class_result, GROUPED_CANDIDATE_ATTR, None) is not None


def get_grouped_view(class_result) -> Optional[GroupedLigandView]:
    return getattr(class_result, GROUPED_CANDIDATE_ATTR, None)
