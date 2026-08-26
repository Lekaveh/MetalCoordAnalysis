from types import SimpleNamespace

from metalCoord.cif.utils import (
    ATOM_ID_1,
    ATOM_ID_2,
    ATOM_ID_3,
    COMP_ID,
    VALUE_ANGLE,
    VALUE_ANGLE_ESD,
)
from metalCoord.service import analysis


def _empty_angles():
    return {
        COMP_ID: [],
        ATOM_ID_1: [],
        ATOM_ID_2: [],
        ATOM_ID_3: [],
        VALUE_ANGLE: [],
        VALUE_ANGLE_ESD: [],
    }


def _component_cycle():
    return [
        ("FE1", "Fe", "A", "FS2", "1"),
        ("S1", "S", "A", "FS2", 1, "", 0),
        ("FE2", "Fe", "A", "FS2", "1"),
        ("S2", "S", "A", "FS2", 1, "", 0),
    ]


def test_tetragon_angles_use_packaged_ring_statistics(monkeypatch):
    cycle = _component_cycle()
    monkeypatch.setattr(analysis, "find_minimal_cycles", lambda _: [cycle])

    selected_angles = {
        "FE1": SimpleNamespace(angle=104.0, std=3.0),
        "FE2": SimpleNamespace(angle=106.0, std=4.0),
    }
    monomer = SimpleNamespace(
        code=("A", "FS2", "1"),
        get_angle=lambda metal, _ligand1, _ligand2: selected_angles[metal],
        get_best_class=lambda _metal: SimpleNamespace(
            coordination=4,
            clazz="tetrahedral",
        ),
    )
    angles = _empty_angles()
    angles[COMP_ID].append("FS2")
    angles[ATOM_ID_1].append("S2")
    angles[ATOM_ID_2].append("FE1")
    angles[ATOM_ID_3].append("S1")
    angles[VALUE_ANGLE].append("999.0")
    angles[VALUE_ANGLE_ESD].append("999.0")

    analysis.update_tetragons("FS2", angles, monomer, cycle)

    by_center = {
        center: (atom1, atom3, value, std)
        for atom1, center, atom3, value, std in zip(
            angles[ATOM_ID_1],
            angles[ATOM_ID_2],
            angles[ATOM_ID_3],
            angles[VALUE_ANGLE],
            angles[VALUE_ANGLE_ESD],
        )
    }
    assert len(angles[COMP_ID]) == 4
    assert by_center["FE1"] == ("S2", "S1", "104.89", "4.13")
    assert by_center["FE2"] == ("S1", "S2", "104.89", "4.13")
    assert by_center["S1"] == ("FE1", "FE2", "74.47", "4.05")
    assert by_center["S2"] == ("FE1", "FE2", "74.47", "4.05")


def test_tetragon_without_packaged_stats_uses_selected_class(monkeypatch):
    cycle = _component_cycle()
    monkeypatch.setattr(analysis, "find_minimal_cycles", lambda _: [cycle])

    selected_angles = {
        "FE1": SimpleNamespace(angle=104.0, std=3.0),
        "FE2": SimpleNamespace(angle=106.0, std=4.0),
    }
    monomer = SimpleNamespace(
        code=("A", "FS2", "1"),
        get_angle=lambda metal, _ligand1, _ligand2: selected_angles[metal],
        get_best_class=lambda _metal: SimpleNamespace(
            coordination=5,
            clazz="trigonal-bipyramid",
        ),
    )
    angles = _empty_angles()

    analysis.update_tetragons("FS2", angles, monomer, cycle)

    by_center = {
        center: (value, std)
        for center, value, std in zip(
            angles[ATOM_ID_2],
            angles[VALUE_ANGLE],
            angles[VALUE_ANGLE_ESD],
        )
    }
    assert by_center["FE1"] == ("104.0", "3.0")
    assert by_center["FE2"] == ("106.0", "4.0")
    assert by_center["S1"] == ("75.0", "5.0")
    assert by_center["S2"] == ("75.0", "5.0")


def test_tetragon_with_external_atom_is_not_written(monkeypatch):
    cycle = _component_cycle()
    cycle[3] = ("SG", "S", "B", "CYS", 42, "", 0)
    monkeypatch.setattr(analysis, "find_minimal_cycles", lambda _: [cycle])

    def fail_if_called(*_args):
        raise AssertionError(
            "external cycles must be rejected before stats lookup"
        )

    monomer = SimpleNamespace(
        code=("A", "FS2", "1"),
        get_angle=fail_if_called,
    )
    angles = _empty_angles()

    analysis.update_tetragons("FS2", angles, monomer, cycle)

    assert all(not values for values in angles.values())


def test_tetragon_without_class_angles_is_not_written(monkeypatch):
    cycle = _component_cycle()
    monkeypatch.setattr(analysis, "find_minimal_cycles", lambda _: [cycle])
    monomer = SimpleNamespace(
        code=("A", "FS2", "1"),
        get_angle=lambda *_args: None,
        get_best_class=lambda _metal: None,
    )
    angles = _empty_angles()

    analysis.update_tetragons("FS2", angles, monomer, cycle)

    assert all(not values for values in angles.values())
