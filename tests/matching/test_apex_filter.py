import gemmi

from metalCoord.analysis.structures import Atom, CifAtom, Ligand


def _atom(name, element, xyz, altloc=""):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    atom.occ = 1
    atom.b_iso = 10
    atom.altloc = altloc if altloc else "\x00"
    return atom


def _model_ligand(metal_spec, donor_specs, base_names=()):
    """Builds a model-path Ligand; donors named in `base_names` become base ligands."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(40, 40, 40, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    metal_atom = _atom(*metal_spec)
    residue.add_atom(metal_atom)
    donor_atoms = []
    for spec in donor_specs:
        donor_atom = _atom(*spec)
        residue.add_atom(donor_atom)
        donor_atoms.append(donor_atom)

    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)

    ligand = Ligand(Atom(metal_atom, residue, chain, None, st))
    for donor_atom in donor_atoms:
        donor = Atom(donor_atom, residue, chain, None, st)
        if donor_atom.name in base_names:
            ligand.add_ligand(donor)
        else:
            ligand.add_extra_ligand(donor)
    return ligand


def _cif_ligand(metal_spec, donor_specs, bonds, atom_class=CifAtom):
    """Builds a CIF-path Ligand whose connectivity comes from `bonds`."""
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom(*metal_spec), residue, chain))
    ligand.nonmetal_bonds = bonds
    for spec in donor_specs:
        ligand.add_ligand(atom_class(_atom(*spec), residue, chain))
    return ligand


def _names(ligand):
    return sorted(atom.atom.name for atom in ligand.all_ligands)


# Carboxylate-like geometry: C at the origin, O-C-O = 122 deg, C-O = 1.25 A, metal on the
# interior bisector 2.45 A from each oxygen -- so the apex is 2.80 A away, farther than both.
CARBOXYLATE = [
    ("O1", "O", (1.0933, 0.606, 0.0)),
    ("C1", "C", (0.0, 0.0, 0.0)),
    ("O2", "O", (-1.0933, 0.606, 0.0)),
]
CARBOXYLATE_METAL = ("CA", "Ca", (0.0, 2.7985, 0.0))


def test_in_plane_apex_is_dropped():
    ligand = _model_ligand(CARBOXYLATE_METAL, CARBOXYLATE)

    removed = ligand.filter_apex_donors()

    assert [atom.atom.name for atom, _, _ in removed] == ["C1"]
    assert ligand.coordination() == 2
    assert _names(ligand) == ["O1", "O2"]

    _, central_angle, tilt = removed[0]
    assert abs(central_angle - 122.0) < 0.5
    assert tilt < 1e-6


def test_apex_above_the_plane_is_kept():
    # Same three atoms, metal lifted out of the O-C-O plane (eta-face like): the apex is
    # still the farthest atom, but the metal is no longer in plane (tilt 36.9 deg).
    ligand = _model_ligand(("CA", "Ca", (0.0, 2.0, 1.5)), CARBOXYLATE)

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_narrow_central_angle_is_kept():
    # O-C-O = 100 deg, metal in plane and the apex farthest -- only the angle fails.
    ligand = _model_ligand(
        ("CA", "Ca", (0.0, 3.0586, 0.0)),
        [
            ("O1", "O", (0.9576, 0.8035, 0.0)),
            ("C1", "C", (0.0, 0.0, 0.0)),
            ("O2", "O", (-0.9576, 0.8035, 0.0)),
        ],
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_central_donor_closer_than_edges_is_kept():
    # Imidazole-like C-N-C at 108 deg with the metal bonded to the central N from the
    # lone-pair side: wide angle and in plane, but the central atom is the closest.
    # Regression guard for the Cd-His pattern seen in 1GKL / 1WB5.
    ligand = _model_ligand(
        ("ZN", "Zn", (0.0, -2.10, 0.0)),
        [
            ("C1", "C", (1.0922, 0.7935, 0.0)),
            ("N1", "N", (0.0, 0.0, 0.0)),
            ("C2", "C", (-1.0922, 0.7935, 0.0)),
        ],
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "C2", "N1"]


def test_group_of_four_donors_is_untouched():
    # The carboxylate apex would qualify, but a fourth bonded donor makes the group wider
    # than three atoms.
    ligand = _model_ligand(CARBOXYLATE_METAL, CARBOXYLATE + [("C2", "C", (0.0, -1.52, 0.0))])

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "C2", "O1", "O2"]


def test_triangle_is_skipped():
    # Apex geometry, but the three donors are mutually bonded -- no unambiguous centre.
    ligand = _cif_ligand(
        CARBOXYLATE_METAL,
        CARBOXYLATE,
        {"O1": ["C1", "O2"], "C1": ["O1", "O2"], "O2": ["O1", "C1"]},
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_overlapping_copies_do_not_form_an_apex():
    # Partial-occupancy copies modelled as separate atoms (the PEP/2PG pattern in 1ONE):
    # the two oxygens coincide, so the triple is degenerate and nothing is removed.
    ligand = _model_ligand(
        CARBOXYLATE_METAL,
        [
            ("O1", "O", (1.0933, 0.606, 0.0)),
            ("C1", "C", (0.0, 0.0, 0.0)),
            ("O2", "O", (1.0933, 0.626, 0.0)),
        ],
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_near_collinear_triple_is_kept():
    ligand = _cif_ligand(
        ("CA", "Ca", (0.0, 6.0, 0.0)),
        [
            ("O1", "O", (1.25, -0.1, 0.0)),
            ("C1", "C", (0.0, 0.0, 0.0)),
            ("O2", "O", (-1.25, -0.1, 0.0)),
        ],
        {"O1": ["C1"], "C1": ["O1", "O2"], "O2": ["C1"]},
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_symmetry_copy_apex_is_kept():
    class _SymmetryAtom(CifAtom):
        @property
        def symmetry(self):
            return 1

    ligand = _cif_ligand(
        CARBOXYLATE_METAL,
        CARBOXYLATE,
        {"O1": ["C1"], "C1": ["O1", "O2"], "O2": ["C1"]},
        atom_class=_SymmetryAtom,
    )

    assert ligand.filter_apex_donors() == []
    assert _names(ligand) == ["C1", "O1", "O2"]


def test_base_donors_are_protected_when_requested():
    ligand = _model_ligand(CARBOXYLATE_METAL, CARBOXYLATE, base_names=("C1",))

    assert ligand.filter_apex_donors(include_base=False) == []
    assert _names(ligand) == ["C1", "O1", "O2"]

    removed = ligand.filter_apex_donors(include_base=True)

    assert [atom.atom.name for atom, _, _ in removed] == ["C1"]
    assert _names(ligand) == ["O1", "O2"]


def test_cif_connectivity_apex_is_dropped():
    ligand = _cif_ligand(
        CARBOXYLATE_METAL,
        CARBOXYLATE,
        {"O1": ["C1"], "C1": ["O1", "O2"], "O2": ["C1"]},
    )

    removed = ligand.filter_apex_donors()

    assert [atom.atom.name for atom, _, _ in removed] == ["C1"]
    assert _names(ligand) == ["O1", "O2"]
