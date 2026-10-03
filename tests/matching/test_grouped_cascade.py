"""Classification-cascade tests for the always-on grouped view.

Cascade (``find_classes_from_structures``):

    D  = classify(S)
    G  = classify(group(S))            # whatever D found
    if D and G are empty:
        S' = clean_the_farthest(S)     # skipped when the CN did not drop
        C  = classify(S')
        G' = classify(group(S'))       # whatever C found

A forced-class ValueError from D is raised only when all branches are empty.
The classificator is faked so each branch's outcome is set explicitly by
(is_grouped_view, coordination).
"""

import gemmi
import numpy as np
import pytest

from metalCoord.analysis import classify as classify_mod
from metalCoord.analysis import grouping as grouping_mod
from metalCoord.analysis.classes import ClassificationResult, idealClasses
from metalCoord.analysis.classify import _candidate_origin
from metalCoord.analysis.grouping import (
    GroupedLigandView,
    create_grouped_view,
    is_grouped_candidate,
    mark_grouped_candidate,
)
from metalCoord.analysis.models import LigandStats
from metalCoord.analysis.structures import Atom, CifAtom, Ligand


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _atom(name, element, xyz):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    atom.occ = 1
    atom.b_iso = 10
    atom.altloc = "\x00"
    return atom


def _site(with_extra=True):
    """CN-5 site: group {A1, A2} + singletons B1, B2 (+ a far extra X1).

    Grouped view: CN 4 (CN 3 once X1 is cleaned). Without X1 the site has CN 4,
    and `clean_the_farthest(free=False)` has nothing to remove.
    """
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom("CU", "Cu", (0, 0, 0)), residue, chain))
    ligand.nonmetal_bonds = {"A1": ["A2"], "A2": ["A1"]}
    for name, element, xyz in (
        ("A1", "N", (2.0, 0.3, 0.0)),
        ("A2", "N", (2.0, -0.3, 0.0)),
        ("B1", "O", (0.0, 2.0, 0.0)),
        ("B2", "O", (0.0, -2.0, 0.0)),
    ):
        ligand.add_ligand(CifAtom(_atom(name, element, xyz), residue, chain))
    if with_extra:
        ligand.add_extra_ligand(
            CifAtom(_atom("X1", "O", (-3.5, 0.0, 0.0)), residue, chain)
        )
    return ligand


def _class_result(clazz):
    coord = idealClasses.get_coordinates(clazz)
    indices = np.array([np.arange(len(coord))])
    return ClassificationResult(clazz, coord, indices, 0.05)


class _FakeClassificator:
    """Yields classes from `table[(is_grouped_view, coordination)]`.

    A table value of "error" raises ValueError (a forced class that does not fit).
    """

    table = {}
    calls = []

    def classify(self, structure, class_name=None):
        key = (isinstance(structure, GroupedLigandView), structure.coordination())
        _FakeClassificator.calls.append(key)
        entry = _FakeClassificator.table.get(key, [])
        if entry == "error":
            raise ValueError("forced class does not fit")
        for clazz in entry:
            yield _class_result(clazz)


class _FakeStrategy:
    """Stand-in for the COD strategy chain: one LigandStats per direct class."""

    def get_stats(self, structure, data, class_result):
        return LigandStats(
            class_result.clazz,
            "",
            np.array([]),
            class_result.proc,
            len(class_result.coord) - 1,
            1,
            "direct",
        )


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(classify_mod, "Classificator", _FakeClassificator)
    monkeypatch.setattr(classify_mod, "strategies", [_FakeStrategy()])
    _FakeClassificator.calls = []

    def run(table, site=None, clazz=None):
        _FakeClassificator.table = table
        results = classify_mod.find_classes_from_structures(
            [site if site is not None else _site()], {}, clazz=clazz
        )
        metals = list(results.metals)
        return metals[0] if metals else None

    return run


def _classes(metal_stats):
    return sorted(ligand.clazz for ligand in metal_stats.ligands)


# --------------------------------------------------------------------------- #
# cascade branches
# --------------------------------------------------------------------------- #
def test_grouped_view_classified_even_when_direct_class_found(cascade):
    metal = cascade(
        {(False, 5): ["trigonal-bipyramid"], (True, 4): ["tetrahedral"]}
    )

    assert _classes(metal) == ["tetrahedral", "trigonal-bipyramid"]
    # the direct class keeps winning: highest coordination first
    assert metal.get_best_class().clazz == "trigonal-bipyramid"
    # no clean branch: D and G both ran on the observed site only
    assert _FakeClassificator.calls == [(False, 5), (True, 4)]


def test_grouped_only_site_skips_clean(cascade):
    metal = cascade({(True, 4): ["tetrahedral"]})

    assert _classes(metal) == ["tetrahedral"]
    assert metal.get_best_class().description.startswith("Grouped fallback")
    assert _FakeClassificator.calls == [(False, 5), (True, 4)]


def test_cleaned_site_also_classifies_its_grouped_view(cascade):
    metal = cascade(
        {(False, 4): ["square-planar"], (True, 3): ["trigonal-planar"]}
    )

    assert _classes(metal) == ["square-planar", "trigonal-planar"]
    assert metal.get_best_class().clazz == "square-planar"
    assert _FakeClassificator.calls == [(False, 5), (True, 4), (False, 4), (True, 3)]


def test_cleaned_grouped_view_rescues_site_when_cleaned_direct_fails(cascade):
    metal = cascade({(True, 3): ["trigonal-planar"]})

    best = metal.get_best_class()
    assert best.clazz == "trigonal-planar"
    assert best.description.startswith("Grouped fallback")
    # the grouped class was built on the cleaned site: X1 is gone everywhere
    payload = best.to_dict()
    names = [entry["ligand"]["name"] for entry in payload["base"] + payload["pdb"]]
    assert "X1" not in names


def test_clean_skipped_when_it_removes_nothing(cascade):
    metal = cascade({}, site=_site(with_extra=False))

    # D (CN 4) and G (CN 3) only: the unchanged site is not classified again
    assert _FakeClassificator.calls == [(False, 4), (True, 3)]
    # no class at all -> covalent statistics
    assert metal.get_best_class().description == "Based on covalent distances"


# --------------------------------------------------------------------------- #
# forced class (--cl)
# --------------------------------------------------------------------------- #
def test_forced_class_error_deferred_to_clean_branch(cascade):
    metal = cascade(
        {(False, 5): "error", (True, 4): "error", (False, 4): ["square-planar"]},
        clazz="square-planar",
    )

    assert metal.get_best_class().clazz == "square-planar"


def test_forced_class_error_raised_when_every_branch_fails(cascade):
    with pytest.raises(ValueError, match="forced class does not fit"):
        cascade(
            {
                (False, 5): "error",
                (True, 4): "error",
                (False, 4): "error",
                (True, 3): "error",
            },
            clazz="linear",
        )


# --------------------------------------------------------------------------- #
# debug origin
# --------------------------------------------------------------------------- #
def test_candidate_origin_labels():
    view = create_grouped_view(_site())
    plain = _class_result("tetrahedral")
    grouped = mark_grouped_candidate(_class_result("tetrahedral"), view)
    assert is_grouped_candidate(grouped)

    assert _candidate_origin(plain, cleaned=False) == "direct"
    assert _candidate_origin(grouped, cleaned=False) == "grouped"
    assert _candidate_origin(plain, cleaned=True) == "cleaned"
    assert _candidate_origin(grouped, cleaned=True) == "cleaned+grouped"


# --------------------------------------------------------------------------- #
# model-graph cache
# --------------------------------------------------------------------------- #
def _model_sites():
    """Two metal sites of one gemmi model, each with a bridged N1-C-N2 pair."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(40, 40, 40, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    sites = []
    for seq, x0 in ((1, 0.0), (2, 20.0)):
        residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(str(seq))
        for atom in (
            _atom("CU", "Cu", (x0, 0, 0)),
            _atom("N1", "N", (x0 + 1.0, 0, 0)),
            _atom("C1", "C", (x0 + 2.4, 0, 0)),
            _atom("N2", "N", (x0 + 3.8, 0, 0)),
        ):
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)

    for residue in st[0]["A"]:
        site = Ligand(Atom(residue["CU"][0], residue, st[0]["A"], None, st))
        site.add_ligand(Atom(residue["N1"][0], residue, st[0]["A"], None, st))
        site.add_ligand(Atom(residue["N2"][0], residue, st[0]["A"], None, st))
        sites.append(site)
    return sites


def test_model_graph_built_once_per_structure(monkeypatch):
    built = []
    original = grouping_mod._graph_from_model

    def counting(structure):
        built.append(structure)
        return original(structure)

    monkeypatch.setattr(grouping_mod, "_graph_from_model", counting)
    cache = {}
    views = [create_grouped_view(site, graph_cache=cache) for site in _model_sites()]

    assert all(view is not None and view.coordination() == 1 for view in views)
    assert len(built) == 1


def test_model_graph_not_cached_without_cache(monkeypatch):
    built = []
    original = grouping_mod._graph_from_model

    def counting(structure):
        built.append(structure)
        return original(structure)

    monkeypatch.setattr(grouping_mod, "_graph_from_model", counting)
    for site in _model_sites():
        create_grouped_view(site)

    assert len(built) == 2
