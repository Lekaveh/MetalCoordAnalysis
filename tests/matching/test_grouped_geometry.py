"""Geometry unit tests for the generalised (centroid) grouped restraints.

Targets the pure math helper ``_centroid_distance`` (Apollonius / generalised
median), the raw ideal-distance lookup ``_ideal_distance_raw`` and the wired
``_grouped_centroid_distance``.  The pure helper is exercised with hand-computed
inputs so the tests are deterministic and independent of the distance DB.
"""

import math

import gemmi
import numpy as np
import pytest

from metalCoord.analysis import stats
from metalCoord.analysis.models import MIN_DISTANCE_STD
from metalCoord.analysis.stats import (
    _MAX_CENTROID_STD,
    _centroid_distance,
    _grouped_centroid_distance,
    _ideal_distance_raw,
)
from metalCoord.analysis.structures import CifAtom, Ligand
from metalCoord.logging import Logger


@pytest.fixture(autouse=True)
def _reset_logger_capture():
    """Keep the process-wide Logger capture state from leaking between tests."""
    yield
    Logger().enable_capture(False, reset=True)


# --------------------------------------------------------------------------- #
# helpers (mirror the tests/matching conventions)
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


def _ligand(metal_element, member_specs):
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")
    ligand = Ligand(CifAtom(_atom("M", metal_element, (0, 0, 0)), residue, chain))
    members = []
    for name, element, xyz in member_specs:
        member = CifAtom(_atom(name, element, xyz), residue, chain)
        ligand.add_ligand(member)
        members.append(member)
    return ligand, members


def _c_matrix(pairs, n):
    """Symmetric (n, n) inter-member distance matrix from {(i, j): value}."""
    c = np.zeros((n, n))
    for (i, j), value in pairs.items():
        c[i, j] = c[j, i] = value
    return c


# --------------------------------------------------------------------------- #
# pure math: _centroid_distance(d, sigma, c) -> (d0, sigma0, R_raw)
# --------------------------------------------------------------------------- #
def test_two_atom_group_matches_median():
    d = np.array([2.0, 2.0])
    sigma = np.array([0.05, 0.05])
    c = _c_matrix({(0, 1): 2.0}, 2)

    d0, sigma0, r_raw = _centroid_distance(d, sigma, c)

    # d0 = 1/2 * sqrt(2*4 + 2*4 - 4) = 1/2 * sqrt(12) = sqrt(3)
    assert d0 == pytest.approx(math.sqrt(3.0), abs=1e-9)
    assert r_raw == pytest.approx(3.0, abs=1e-9)


def test_single_atom_group_degrades_to_ideal():
    d0, sigma0, r_raw = _centroid_distance(
        np.array([2.0]), np.array([0.05]), np.zeros((1, 1))
    )
    assert d0 == pytest.approx(2.0, abs=1e-12)
    assert sigma0 == pytest.approx(0.05, abs=1e-12)


def test_collinear_members_collapse_to_zero():
    # two members d=1 on opposite sides of M, observed c=2 -> centroid at metal
    d0, sigma0, r_raw = _centroid_distance(
        np.array([1.0, 1.0]), np.array([0.05, 0.05]), _c_matrix({(0, 1): 2.0}, 2)
    )
    assert d0 == pytest.approx(0.0, abs=1e-12)
    assert r_raw == pytest.approx(0.0, abs=1e-12)
    # zero-distance sigma path must not divide by zero
    assert sigma0 == MIN_DISTANCE_STD


def test_three_atom_group_general_formula():
    d = np.array([2.0, 2.0, 2.0])
    sigma = np.array([0.05, 0.05, 0.05])
    c = _c_matrix({(0, 1): 2.0, (0, 2): 2.0, (1, 2): 2.0}, 3)

    d0, sigma0, r_raw = _centroid_distance(d, sigma, c)

    expected_r = 12.0 / 3.0 - 12.0 / 9.0  # (1/n)sum d^2 - (1/n^2)sum c^2
    assert r_raw == pytest.approx(expected_r, abs=1e-9)
    assert d0 == pytest.approx(math.sqrt(expected_r), abs=1e-9)


def test_slightly_negative_radicand_clamped():
    # choose c so R_raw in [-1e-6, 0): R = 1 - c^2/4
    c12 = math.sqrt(4.0 + 2e-6)  # -> R = -5e-7
    d0, sigma0, r_raw = _centroid_distance(
        np.array([1.0, 1.0]), np.array([0.05, 0.05]), _c_matrix({(0, 1): c12}, 2)
    )
    assert d0 == 0.0
    assert -1e-6 <= r_raw < 0.0


def test_materially_negative_radicand_clamps_to_zero():
    # observed spread far wider than ideal bond lengths allow -> R < -eps
    d0, sigma0, r_raw = _centroid_distance(
        np.array([1.0, 1.0]), np.array([0.05, 0.05]), _c_matrix({(0, 1): 2.5}, 2)
    )
    assert d0 == 0.0
    assert r_raw < -1e-6


def test_nan_input_returns_none():
    d0, sigma0, r_raw = _centroid_distance(
        np.array([np.nan, 2.0]), np.array([0.05, 0.05]), _c_matrix({(0, 1): 2.0}, 2)
    )
    assert d0 is None and sigma0 is None and r_raw is None


def test_near_degenerate_sigma_is_capped():
    # d0 just above the zero-guard: the propagated sigma would explode (~70 A),
    # so it must be capped at _MAX_CENTROID_STD rather than emitted raw.
    c12 = math.sqrt(16.0 - 4e-6)  # r_raw ~ 1e-6 -> d0 ~ 0.001
    d0, sigma0, r_raw = _centroid_distance(
        np.array([2.0, 2.0]), np.array([0.05, 0.05]), _c_matrix({(0, 1): c12}, 2)
    )
    assert 1e-9 < d0 < 1e-2  # propagation branch, not the zero-distance floor
    assert sigma0 == pytest.approx(_MAX_CENTROID_STD)


def test_sigma_propagation():
    d = np.array([2.0, 2.0])
    sigma = np.array([0.1, 0.1])
    c = _c_matrix({(0, 1): 2.0}, 2)

    d0, sigma0, _ = _centroid_distance(d, sigma, c)

    n = 2
    expected = math.sqrt(sum((d[i] / (n * d0)) ** 2 * sigma[i] ** 2 for i in range(n)))
    assert sigma0 == pytest.approx(expected, abs=1e-9)
    assert sigma0 == pytest.approx(0.081649, abs=1e-5)


# --------------------------------------------------------------------------- #
# raw ideal-distance lookup: _ideal_distance_raw(metal_elem, ligand_elem)
# --------------------------------------------------------------------------- #
def test_ideal_distance_raw_uses_db(monkeypatch):
    monkeypatch.setattr(stats.DB, "get_distance_stats", lambda m, l: (2.13, 0.047, 12))
    dist, std = _ideal_distance_raw("Cu", "N")
    assert dist == pytest.approx(2.13)
    assert std == pytest.approx(0.047)


def test_ideal_distance_raw_covalent_fallback(monkeypatch):
    monkeypatch.setattr(stats.DB, "get_distance_stats", lambda m, l: (0, 0, 0))
    dist, std = _ideal_distance_raw("Cu", "N")
    expected = gemmi.Element("N").covalent_r + gemmi.Element("Cu").covalent_r
    assert dist == pytest.approx(expected)
    assert std == pytest.approx(0.2)


def test_ideal_distance_raw_nan_std_floored(monkeypatch):
    monkeypatch.setattr(
        stats.DB, "get_distance_stats", lambda m, l: (2.1, float("nan"), 1)
    )
    dist, std = _ideal_distance_raw("Cu", "N")
    assert dist == pytest.approx(2.1)
    assert std == MIN_DISTANCE_STD


# --------------------------------------------------------------------------- #
# wired helper: _grouped_centroid_distance(structure, members)
# --------------------------------------------------------------------------- #
def test_grouped_centroid_distance_wires_positions(monkeypatch):
    monkeypatch.setattr(stats, "_ideal_distance_raw", lambda m, l: (2.0, 0.05))
    ligand, members = _ligand(
        "Cu", [("A1", "N", (2.0, 0.0, 0.0)), ("A2", "N", (2.0, 2.0, 0.0))]
    )
    # observed c_12 = 2.0, ideal d = 2.0 -> d0 = sqrt(3)
    d0, sigma0 = _grouped_centroid_distance(ligand, members)
    assert d0 == pytest.approx(math.sqrt(3.0), abs=1e-6)
    assert sigma0 >= MIN_DISTANCE_STD


def test_grouped_centroid_distance_uses_covalent_when_db_empty(monkeypatch):
    monkeypatch.setattr(stats.DB, "get_distance_stats", lambda m, l: (0, 0, 0))
    ligand, members = _ligand("Cu", [("A1", "N", (2.0, 0.0, 0.0))])
    d0, sigma0 = _grouped_centroid_distance(ligand, members)
    cov = gemmi.Element("N").covalent_r + gemmi.Element("Cu").covalent_r
    assert d0 == pytest.approx(cov, abs=1e-6)  # n == 1 -> d0 == d1


def test_grouped_centroid_distance_material_inconsistency_warns(monkeypatch):
    monkeypatch.setattr(stats, "_ideal_distance_raw", lambda m, l: (1.0, 0.05))
    ligand, members = _ligand(
        "Cu", [("A1", "N", (1.0, 0.0, 0.0)), ("A2", "N", (-1.5, 0.0, 0.0))]
    )
    logger = Logger()
    logger.enable_capture(True, reset=True)
    mark = logger.mark()

    d0, sigma0 = _grouped_centroid_distance(ligand, members)

    assert d0 == 0.0
    records = logger.records_since(mark)
    assert any(r["level"] == "WARNING" for r in records)


def test_grouped_centroid_distance_nan_skips_and_warns(monkeypatch):
    monkeypatch.setattr(stats, "_ideal_distance_raw", lambda m, l: (float("nan"), 0.05))
    ligand, members = _ligand(
        "Cu", [("A1", "N", (1.0, 0.0, 0.0)), ("A2", "N", (2.0, 0.0, 0.0))]
    )
    logger = Logger()
    logger.enable_capture(True, reset=True)
    mark = logger.mark()

    d0, sigma0 = _grouped_centroid_distance(ligand, members)

    assert d0 is None and sigma0 is None
    records = logger.records_since(mark)
    assert any(r["level"] == "WARNING" for r in records)
