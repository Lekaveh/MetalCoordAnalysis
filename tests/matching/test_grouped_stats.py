"""Stats-builder tests for the generalised (centroid) grouped fallback.

A collapsed multi-atom vertex is emitted as a centroid pseudo-atom in the new
``grouped_base`` / ``grouped_angles`` JSON keys, while the per-atom
``_chem_comp_bond`` connectivity (``base`` / ``order``) stays populated.  Angle
restraints route by vertex pair: any pair touching a true group goes to
``grouped_angles`` (polymorphic endpoints), a singleton-singleton pair stays in
the standard ``angles``.
"""

import gemmi
import numpy as np
import pytest

from metalCoord.analysis.classes import ClassificationResult, idealClasses
from metalCoord.analysis.classify import find_classes_from_structures
from metalCoord.analysis.grouping import create_grouped_view, mark_grouped_candidate
from metalCoord.analysis.models import MIN_DISTANCE_STD
from metalCoord.analysis.stats import create_grouped_ligand_stats
from metalCoord.analysis.structures import CifAtom, Ligand


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def _trigonal_stats():
    """One true group {A1, A2} plus singletons {B1}, {C1} -> trigonal-planar."""
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
    return ligand, create_grouped_ligand_stats(ligand, class_result)


def _centroid_endpoint(angle_dict):
    for key in ("ligand1", "ligand2"):
        if angle_dict[key].get("kind") == "centroid":
            return angle_dict[key]
    return None


def _plain_endpoint(angle_dict):
    for key in ("ligand1", "ligand2"):
        if angle_dict[key].get("kind") != "centroid":
            return angle_dict[key]
    return None


# --------------------------------------------------------------------------- #
# grouped distances (grouped_base) -- true groups only
# --------------------------------------------------------------------------- #
def test_grouped_distance_count_equals_true_group_count():
    _, stats = _trigonal_stats()
    payload = stats.to_dict()
    # only the one true group {A1, A2} yields a centroid distance
    assert len(payload["grouped_base"]) == 1


def test_grouped_distance_members_are_real_atoms():
    _, stats = _trigonal_stats()
    grouped_bond = stats.to_dict()["grouped_base"][0]
    ligand = grouped_bond["ligand"]

    assert ligand["kind"] == "centroid"
    assert ligand["n"] == 2
    assert ligand["name"].startswith("Q")
    member_names = {m["name"] for m in ligand["members"]}
    assert member_names == {"A1", "A2"}
    # members carry full atom dicts, not just names
    assert all("element" in m and "residue" in m for m in ligand["members"])
    assert grouped_bond["distance"] > 0
    assert grouped_bond["std"] >= MIN_DISTANCE_STD


# --------------------------------------------------------------------------- #
# grouped angles -- every pair with >= 1 group; polymorphic endpoints
# --------------------------------------------------------------------------- #
def test_grouped_angles_pairs_with_at_least_one_group():
    _, stats = _trigonal_stats()
    grouped_angles = stats.to_dict()["grouped_angles"]

    # group-B1 and group-C1 -> two grouped angles; B1-C1 is NOT here
    assert len(grouped_angles) == 2

    singletons = set()
    for entry in grouped_angles:
        assert _centroid_endpoint(entry) is not None
        plain = _plain_endpoint(entry)
        assert plain is not None
        singletons.add(plain["name"])
        assert entry["angle"] == pytest.approx(120.0, abs=1.0)
    assert singletons == {"B1", "C1"}


def test_singleton_singleton_angle_in_standard_angles():
    _, stats = _trigonal_stats()
    standard = list(stats.angles)

    assert len(standard) == 1
    pair = {standard[0].ligand1.name, standard[0].ligand2.name}
    assert pair == {"B1", "C1"}
    assert len(stats.to_dict()["angles"]) == 1


# --------------------------------------------------------------------------- #
# connectivity + JSON shape
# --------------------------------------------------------------------------- #
def test_standard_bonds_present_grouped_populated():
    _, stats = _trigonal_stats()
    payload = stats.to_dict()

    # per-atom connectivity for all four real atoms, no centroid leakage
    assert len(payload["base"]) == 4
    assert len(payload["order"]) == 4
    assert {b["ligand"]["name"] for b in payload["base"]} == {"A1", "A2", "B1", "C1"}
    assert all(not b["ligand"]["name"].startswith("Q") for b in payload["base"])
    assert all(not atom["name"].startswith("Q") for atom in payload["order"])

    assert len(payload["grouped_base"]) == 1
    assert len(payload["grouped_angles"]) == 2


def test_to_dict_exposes_grouped_keys():
    _, stats = _trigonal_stats()
    payload = stats.to_dict()
    assert "grouped_base" in payload
    assert "grouped_angles" in payload


def test_metadata_preserved():
    _, stats = _trigonal_stats()
    payload = stats.to_dict()
    assert payload["coordination"] == 3
    assert payload["count"] == -1
    assert payload["description"] == "Grouped fallback (1 groups collapsed)"


# --------------------------------------------------------------------------- #
# fixed-class ValueError path (two groups collapsed) -- end to end
# --------------------------------------------------------------------------- #
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

    # connectivity retains all four real atoms
    assert len(payload["base"]) == 4
    assert len(payload["order"]) == 4

    # two centroid distances, one group-group angle, no standard angles
    assert len(payload["grouped_base"]) == 2
    assert len(payload["grouped_angles"]) == 1
    assert len(payload["angles"]) == 0

    grouped_angle = payload["grouped_angles"][0]
    assert grouped_angle["ligand1"]["kind"] == "centroid"
    assert grouped_angle["ligand2"]["kind"] == "centroid"
    assert grouped_angle["angle"] == pytest.approx(180.0, abs=1.0)
