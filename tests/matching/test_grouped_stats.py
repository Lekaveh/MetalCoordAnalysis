import gemmi
import numpy as np

from metalCoord.analysis.classes import ClassificationResult, idealClasses
from metalCoord.analysis.classify import find_classes_from_structures
from metalCoord.analysis.grouping import create_grouped_view, mark_grouped_candidate
from metalCoord.analysis.stats import create_grouped_ligand_stats
from metalCoord.analysis.structures import CifAtom, Ligand


def _adjacency(*edges):
    graph = {}
    for atom1, atom2 in edges:
        graph.setdefault(atom1, []).append(atom2)
        graph.setdefault(atom2, []).append(atom1)
    return graph


def _atom(name, element, xyz):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    atom.occ = 1
    atom.b_iso = 10
    atom.altloc = "\x00"
    return atom


def _ligand(atom_specs, bonds):
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom("CU", "Cu", (0, 0, 0)), residue, chain))
    ligand.nonmetal_bonds = bonds
    for name, element, xyz in atom_specs:
        ligand.add_ligand(CifAtom(_atom(name, element, xyz), residue, chain))
    return ligand


def _class_result(clazz):
    coord = idealClasses.get_coordinates(clazz)
    indices = np.array([np.arange(len(coord))])
    return ClassificationResult(clazz, coord, indices, 0.05)


def test_grouped_stats_map_ideal_angles_to_inter_group_real_atoms():
    ligand = _ligand(
        [
            ("A1", "N", (1.0, 0.0, 0.0)),
            ("A2", "N", (1.0, 0.2, 0.0)),
            ("B1", "O", (0.0, 1.0, 0.0)),
            ("C1", "O", (-1.0, 0.0, 0.0)),
        ],
        _adjacency(("A1", "A2")),
    )
    grouped = create_grouped_view(ligand)
    class_result = mark_grouped_candidate(_class_result("trigonal-planar"), grouped)

    stats = create_grouped_ligand_stats(ligand, class_result)

    payload = stats.to_dict()
    angle_pairs = {
        frozenset([angle.ligand1.name, angle.ligand2.name]) for angle in stats.angles
    }

    assert payload["coordination"] == 3
    assert payload["count"] == -1
    assert payload["description"] == "Grouped fallback (1 groups collapsed)"
    assert len(payload["base"]) == 4
    assert len(payload["order"]) == 4
    assert all(not bond["ligand"]["name"].startswith("G") for bond in payload["base"])
    assert all(not atom["name"].startswith("G") for atom in payload["order"])
    assert len(list(stats.angles)) == 5
    assert frozenset(["A1", "A2"]) not in angle_pairs
    assert frozenset(["A1", "B1"]) in angle_pairs
    assert frozenset(["A2", "C1"]) in angle_pairs


def test_grouped_fallback_handles_fixed_class_value_error():
    ligand = _ligand(
        [
            ("A1", "N", (1.0, 0.1, 0.0)),
            ("A2", "N", (1.0, -0.1, 0.0)),
            ("B1", "N", (-1.0, 0.1, 0.0)),
            ("B2", "N", (-1.0, -0.1, 0.0)),
        ],
        _adjacency(("A1", "A2"), ("B1", "B2")),
    )

    results = find_classes_from_structures([ligand], {}, clazz="linear")
    metals = list(results.metals)
    best = metals[0].get_best_class()
    payload = best.to_dict()

    assert len(metals) == 1
    assert best.clazz == "linear"
    assert best.coordination == 2
    assert best.description == "Grouped fallback (2 groups collapsed)"
    assert len(payload["base"]) == 4
    assert len(payload["order"]) == 4
