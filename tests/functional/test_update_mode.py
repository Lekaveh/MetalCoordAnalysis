import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import gemmi
import pytest


# CIF Categories
ANGLE_CATEGORY = "_chem_comp_angle"
BOND_CATEGORY = "_chem_comp_bond"
ATOM_CATEGORY = "_chem_comp_atom"
ACEDRG_CATEGORY = "_acedrg_chem_comp_descriptor"

VALUE_DIST = "value_dist"
VALUE_DIST_ESD = "value_dist_esd"
VALUE_ANGLE = "value_angle"
VALUE_ANGLE_ESD = "value_angle_esd"
ATOM_ID_1 = "atom_id_1"
ATOM_ID_2 = "atom_id_2"
ATOM_ID_3 = "atom_id_3"
VALUE_DIST_NUCLEUS = "value_dist_nucleus"
VALUE_DIST_NUCLEUS_ESD = "value_dist_nucleus_esd"


is_windows = sys.platform.startswith('win')

# TestCase definition
class UpdateModeTestCase(NamedTuple):
    """
    A NamedTuple representing a test case for the update mode functionality.

    Attributes:
        name (str): The name of the test case.
        input (str): The input data for the test case.
        output (str): The expected output data for the test case.
        pdb (str): The PDB (Protein Data Bank) file associated with the test case.
        description (str): A brief description of the test case.
    """
    name: str
    input: str
    output: str
    pdb: str
    description: str


tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define test cases
TEST_CASES = [
    UpdateModeTestCase(
        "SF4",
        os.path.join(tests_dir, str(Path("./data/dicts/SF4.cif"))),
        os.path.join(tests_dir, str(Path('./data/results/SF4.cif'))),
        "5d8v",
        "IRON/SULFUR CLUSTER"
    ),
    UpdateModeTestCase(
        "6SG",
        os.path.join(tests_dir, str(Path("./data/dicts/6SG.cif"))),
        os.path.join(tests_dir, str(Path('./data/results/6SG.cif'))),
        "5l6x",
        "S-[N-(ferrocenylmethyl)carbamoylmethyl]-glutathione"
    ),
]


@pytest.fixture(params=TEST_CASES)
def test_case(request: Any) -> UpdateModeTestCase:
    """
    A test case function that retrieves the parameter from the request object.

    Args:
        request (Any): The request object containing the test parameters.

    Returns:
        UpdateModeTestCase: The test case parameter extracted from the request.
    """
    return request.param


@pytest.fixture
def temp_dir(test_case: UpdateModeTestCase, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Create a temporary directory for the given test case.

    Args:
        test_case (UpdateModeTestCase): The test case instance containing the name attribute.
        tmp_path_factory (pytest.TempPathFactory): The pytest factory for creating temporary directories.

    Returns:
        Path: The path to the newly created temporary directory.
    """
    return tmp_path_factory.mktemp(f"data_{test_case.name}_{os.urandom(8).hex()}")


@pytest.fixture
def cli_output(temp_dir: Path, test_case: UpdateModeTestCase) -> Path:
    """
    Executes the 'metalCoord update' command with the provided test case parameters
    and verifies the output.
    Args:
        temp_dir (Path): The temporary directory where the output CIF file will be stored.
        test_case (UpdateModeTestCase): An instance of UpdateModeTestCase containing the name,
        input file path, output file path, and pdb file path.
    Returns:
        Path: The path to the generated CIF file.
    Raises:
        AssertionError: If the CLI command fails or the generated CIF file is not found.
    """
    name = test_case.name
    input = test_case.input
    pdb = test_case.pdb
    output_path = os.path.join(temp_dir, f"{name}.cif")

    test_args = [
        'metalCoord', 'update',
        '-i', input,
        '-p', pdb,
        '-o', output_path
    ]

    result = subprocess.run(
        test_args, capture_output=True, text=True, check=True, shell=is_windows
    )

    assert result.returncode == 0, f"CLI command failed for {name}"
    assert Path(output_path).exists(
    ), f"Generated CIF file not found for {name}"

    return output_path

# Helper functions to extract distances and angles from CIF


def get_block_from_cif(path):
    """
    Extracts a block from a CIF (Crystallographic Information File) based on specific naming patterns.

    Args:
        path (str): The file path to the CIF file.

    Returns:
        gemmi.cif.Block: The block object extracted from the CIF file.

    Raises:
        ValueError: If no block matching the required naming patterns is found in the CIF file.

    """
    doc = gemmi.cif.read_file(path)

    name = None
    for block in doc:
        matches = re.findall(r"^(?:comp_)?([A-Za-z0-9]{3,}$)", block.name)
        if matches:
            name = matches[0]
            if name == 'list':
                continue
            break

    if not name:
        raise ValueError(
            "No block found for <name>|comp_<name>. Please check the CIF file.")

    block = doc.find_block(f"comp_{name}") if doc.find_block(
        f"comp_{name}") is not None else doc.find_block(f"{name}")

    if block is None:
        raise ValueError(
            f"No block found for {name}|comp_{name}. Please check the CIF file.")

    return block


def get_distances_from_cif(block):
    """
    Extracts bond distance information from a CIF (Crystallographic Information File) block.

    Args:
        block: A CIF block object that contains crystallographic data.

    Returns:
        A list of dictionaries, each containing bond distance information with the following keys:
            - ATOM_ID_1: The identifier of the first atom in the bond.
            - ATOM_ID_2: The identifier of the second atom in the bond.
            - VALUE_DIST: The bond distance value.
            - VALUE_DIST_ESD: The estimated standard deviation of the bond distance (optional).
            - VALUE_DIST_NUCLEUS: The bond distance value for the nucleus (optional).
            - VALUE_DIST_NUCLEUS_ESD: The estimated standard deviation of the bond distance for the nucleus (optional).
    """
    distances = []
    bonds = block.get_mmcif_category(BOND_CATEGORY)
    if bonds:
        atom_id_1_list = bonds[ATOM_ID_1]
        atom_id_2_list = bonds[ATOM_ID_2]
        value_dist_list = bonds[VALUE_DIST]
        value_dist_esd_list = bonds.get(
            VALUE_DIST_ESD, [None] * len(atom_id_1_list))
        value_dist_nucleus_list = bonds.get(
            VALUE_DIST_NUCLEUS, [None] * len(atom_id_1_list))
        value_dist_nucleus_esd_list = bonds.get(
            VALUE_DIST_NUCLEUS_ESD, [None] * len(atom_id_1_list))

        for atom_id_1, atom_id_2, value_dist, value_dist_esd, value_dist_nucleus, value_dist_nucleus_esd in zip(
            atom_id_1_list,
            atom_id_2_list,
            value_dist_list,
            value_dist_esd_list,
            value_dist_nucleus_list,
            value_dist_nucleus_esd_list
        ):
            distances.append({
                ATOM_ID_1: atom_id_1,
                ATOM_ID_2: atom_id_2,
                VALUE_DIST: float(value_dist),
                VALUE_DIST_ESD: float(value_dist_esd) if value_dist_esd else None,
                VALUE_DIST_NUCLEUS: float(value_dist_nucleus) if value_dist_nucleus else None,
                VALUE_DIST_NUCLEUS_ESD: float(
                    value_dist_nucleus_esd) if value_dist_nucleus_esd else None
            })
    return distances


def get_angles_from_cif(block):
    """
    Extracts angle information from a CIF (Crystallographic Information File) block.

    Args:
        block: A CIF block object that contains crystallographic data.

    Returns:
        A list of dictionaries, each containing the following keys:
            - ATOM_ID_1: The identifier of the first atom in the angle.
            - ATOM_ID_2: The identifier of the second atom in the angle.
            - ATOM_ID_3: The identifier of the third atom in the angle.
            - VALUE_ANGLE: The value of the angle in degrees.
            - VALUE_ANGLE_ESD: The estimated standard deviation of the angle value, if available; otherwise, None.
    """
    angles = []
    angle_data = block.get_mmcif_category(ANGLE_CATEGORY)
    if angle_data:
        atom_id_1_list = angle_data[ATOM_ID_1]
        atom_id_2_list = angle_data[ATOM_ID_2]
        atom_id_3_list = angle_data[ATOM_ID_3]
        value_angle_list = angle_data[VALUE_ANGLE]
        value_angle_esd_list = angle_data.get(
            VALUE_ANGLE_ESD, [None] * len(atom_id_1_list))

        for atom_id_1, atom_id_2, atom_id_3, value_angle, value_angle_esd in zip(
            atom_id_1_list,
            atom_id_2_list,
            atom_id_3_list,
            value_angle_list,
            value_angle_esd_list
        ):
            angles.append({
                ATOM_ID_1: atom_id_1,
                ATOM_ID_2: atom_id_2,
                ATOM_ID_3: atom_id_3,
                VALUE_ANGLE: float(value_angle),
                VALUE_ANGLE_ESD: float(
                    value_angle_esd) if value_angle_esd else None
            })
    return angles

# The pytest test function


def test_compare_cif_files(cli_output: Path, test_case: UpdateModeTestCase):
    """
    Compare the distances and angles between atoms in two CIF files.
    This function loads the expected CIF file and the generated CIF file, extracts
    the distances and angles between atoms, and compares them to ensure they match
    within a specified tolerance.
    Args:
        cli_output (Path): Path to the generated CIF file.
        test_case (TestCase): Test case containing the path to the expected CIF file.
    Raises:
        AssertionError: If there is a mismatch in the number of distances or angles,
                        or if any distance or angle values differ beyond the specified tolerance.
    """
    # Load the expected CIF file
    expected_block = get_block_from_cif(test_case.output)

    # Load the generated CIF file
    generated_block = get_block_from_cif(cli_output)

    # Extract distances
    distances1 = get_distances_from_cif(expected_block)
    distances2 = get_distances_from_cif(generated_block)

    # Compare distances
    assert len(distances1) == len(
        distances2), "Mismatch in number of distances."

    for d in distances2:
        assert d[VALUE_DIST] is not None, f"Distance is missing for bond {d[ATOM_ID_1]} - {d[ATOM_ID_2]}"
        assert d[VALUE_DIST_ESD] is not None, f"Distance ESD is missing for bond {d[ATOM_ID_1]} - {d[ATOM_ID_2]}"

    for d1 in distances1:
        match = next(
            (
                d2
                for d2 in distances2
                if (
                    (d1[ATOM_ID_1] == d2[ATOM_ID_1]
                     and d1[ATOM_ID_2] == d2[ATOM_ID_2])
                    or (d1[ATOM_ID_1] == d2[ATOM_ID_2] and d1[ATOM_ID_2] == d2[ATOM_ID_1])
                )
            ),
            None,
        )

        assert match is not None, (
            f"Bond {d1[ATOM_ID_1]} - {d1[ATOM_ID_2]} not found in second CIF file."
        )

        assert abs(d1[VALUE_DIST] - match[VALUE_DIST])/d1[VALUE_DIST] < 0.05, (
            f"Distances differ for bond {d1[ATOM_ID_1]} - {d1[ATOM_ID_2]}. The difference is more than 5%: "
            f"{d1[VALUE_DIST]} vs {match[VALUE_DIST]}"
        )

        if d1[VALUE_DIST_ESD] is not None and match[VALUE_DIST_ESD] is not None:
            assert abs(d1[VALUE_DIST_ESD] - match[VALUE_DIST_ESD])/d1[VALUE_DIST_ESD] < 0.1, (
                f"Distance ESDs differ for bond {d1[ATOM_ID_1]} - {d1[ATOM_ID_2]}. The difference is more than 10%: "
                f"{d1[VALUE_DIST_ESD]} vs {match[VALUE_DIST_ESD]}"
            )

        if d1[VALUE_DIST_NUCLEUS] is not None and match[VALUE_DIST_NUCLEUS] is not None:
            assert abs(d1[VALUE_DIST_NUCLEUS] - match[VALUE_DIST_NUCLEUS])/d1[VALUE_DIST_NUCLEUS] < 0.05, (
                f"Nucleus distances differ for bond {d1[ATOM_ID_1]} - {d1[ATOM_ID_2]}. The difference is more than 5%: "
                f"{d1[VALUE_DIST_NUCLEUS]} vs {match[VALUE_DIST_NUCLEUS]}"
            )

        if (
            d1[VALUE_DIST_NUCLEUS_ESD] is not None
            and match[VALUE_DIST_NUCLEUS_ESD] is not None
        ):
            assert abs(d1[VALUE_DIST_NUCLEUS_ESD] - match[VALUE_DIST_NUCLEUS_ESD])/d1[VALUE_DIST_NUCLEUS_ESD] < 0.1, (
                f"Nucleus distance ESDs differ for bond {d1[ATOM_ID_1]} - {d1[ATOM_ID_2]}: "
                f"{d1[VALUE_DIST_NUCLEUS_ESD]} vs {match[VALUE_DIST_NUCLEUS_ESD]}. The difference is more than 10%: "
            )

    # Extract angles
    angles1 = get_angles_from_cif(expected_block)
    angles2 = get_angles_from_cif(generated_block)

    # Compare angles
    assert len(angles1) == len(angles2), "Mismatch in number of angles."
    for a in angles2:
        assert a[VALUE_ANGLE] is not None, (
            f"Angle is missing for angle {a[ATOM_ID_1]} - {a[ATOM_ID_2]} - {a[ATOM_ID_3]}"
        )
        assert a[VALUE_ANGLE_ESD] is not None, (
            f"Angle ESD is missing for angle {a[ATOM_ID_1]} - {a[ATOM_ID_2]} - {a[ATOM_ID_3]}"
        )
        assert a[VALUE_ANGLE_ESD] >= 1, (
            f"Angle ESD is too low for angle {a[ATOM_ID_1]} - {a[ATOM_ID_2]} - {a[ATOM_ID_3]}: {a[VALUE_ANGLE_ESD]}"
        )

    for a1 in angles1:
        match = next(
            (
                a2
                for a2 in angles2
                if (
                    (a1[ATOM_ID_1] == a2[ATOM_ID_1] and a1[ATOM_ID_2] ==
                     a2[ATOM_ID_2] and a1[ATOM_ID_3] == a2[ATOM_ID_3])
                    or (a1[ATOM_ID_1] == a2[ATOM_ID_3] and a1[ATOM_ID_2] == a2[ATOM_ID_2] and a1[ATOM_ID_3] == a2[ATOM_ID_1])
                )
            ),
            None,
        )
        assert match is not None, (
            f"Angle {a1[ATOM_ID_1]}-{a1[ATOM_ID_2]}-{a1[ATOM_ID_3]} not found in second CIF file."
        )
        assert abs(a1[VALUE_ANGLE] - match[VALUE_ANGLE]) < 0.1, (
            f"Angles differ for {a1[ATOM_ID_1]}-{a1[ATOM_ID_2]}-{a1[ATOM_ID_3]}: {a1[VALUE_ANGLE]} vs {match[VALUE_ANGLE]}"
        )
        if a1[VALUE_ANGLE_ESD] is not None and match[VALUE_ANGLE_ESD] is not None:
            assert abs(a1[VALUE_ANGLE_ESD] - match[VALUE_ANGLE_ESD]) / a1[VALUE_ANGLE_ESD] < 0.1, (
                f"Angle ESDs differ for {a1[ATOM_ID_1]}-{a1[ATOM_ID_2]}-{a1[ATOM_ID_3]}: "
                f"{a1[VALUE_ANGLE_ESD]} vs {match[VALUE_ANGLE_ESD]}. The difference is more than 10%."
            )
