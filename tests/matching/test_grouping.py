import gemmi
import numpy as np

from metalCoord.analysis.grouping import create_grouped_view
from metalCoord.analysis.structures import Atom, CifAtom, Ligand
from metalCoord.cif.utils import (
    ATOM_ID,
    ATOM_ID_1,
    ATOM_ID_2,
    TYPE_SYMBOL,
    get_nonmetal_bonds,
)


def _adjacency(*edges):
    graph = {}
    for atom1, atom2 in edges:
        graph.setdefault(atom1, []).append(atom2)
        graph.setdefault(atom2, []).append(atom1)
    return graph


def _atom(name, element, xyz, altloc=""):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    atom.occ = 1
    atom.b_iso = 10
    atom.altloc = altloc if altloc else "\x00"
    return atom


def _cif_ligand(atom_specs, bonds):
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom("CU", "Cu", (0, 0, 0)), residue, chain))
    ligand.nonmetal_bonds = bonds
    for name, element, xyz, altloc in atom_specs:
        ligand.add_ligand(CifAtom(_atom(name, element, xyz, altloc), residue, chain))
    return ligand


def test_get_nonmetal_bonds_excludes_metal_edges():
    atoms = {
        ATOM_ID: ["CU1", "CU2", "N1", "C1", "O1"],
        TYPE_SYMBOL: ["Cu", "Cu", "N", "C", "O"],
    }
    bonds = {
        ATOM_ID_1: ["CU1", "CU1", "N1", "N1", "CU1"],
        ATOM_ID_2: ["N1", "CU2", "C1", "O1", "CU2"],
    }

    assert get_nonmetal_bonds(atoms, bonds) == {
        "N1": ["C1", "O1"],
        "C1": ["N1"],
        "O1": ["N1"],
    }


def test_cif_grouping_accepts_paths_up_to_three_edges():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("O1", "O", (3, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "C2"), ("C2", "O1")),
    )

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "O1"]
    np.testing.assert_allclose(view.get_coord()[1], [2, 0, 0])


def test_cif_grouping_rejects_paths_longer_than_three_edges():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("O1", "O", (5, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "O1")),
    )

    assert create_grouped_view(ligand) is None


def test_cif_grouping_can_collapse_three_ligands_into_one_centroid():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("N2", "N", (2, 0, 0), ""),
            ("N3", "N", (3, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "N2"), ("N2", "C2"), ("C2", "N3")),
    )

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "N2", "N3"]
    np.testing.assert_allclose(view.get_coord()[1], [2, 0, 0])


def test_cif_grouping_keeps_incompatible_altlocs_separate():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), "A"),
            ("N2", "N", (2, 0, 0), "B"),
        ],
        _adjacency(("N1", "N2")),
    )

    assert create_grouped_view(ligand) is None


def test_model_grouping_uses_coordinate_bond_graph_with_helpers():
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    metal_atom = _atom("CU", "Cu", (0, 0, 0))
    ligand1 = _atom("N1", "N", (1.0, 0, 0))
    helper = _atom("C1", "C", (2.4, 0, 0))
    ligand2 = _atom("N2", "N", (3.8, 0, 0))
    for atom in (metal_atom, ligand1, helper, ligand2):
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)

    ligand = Ligand(Atom(metal_atom, residue, chain, None, st))
    ligand.add_ligand(Atom(ligand1, residue, chain, None, st))
    ligand.add_ligand(Atom(ligand2, residue, chain, None, st))

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "N2"]
