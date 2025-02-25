import json
import os
import subprocess
from pathlib import Path
from typing import Any, List, NamedTuple

import pytest


class StatsModeTestCase(NamedTuple):
    """
    A named tuple representing a test case for metal coordination analysis.

    Attributes:
        ligand_name (str): The name of the ligand involved in the test case.
        model (str): The model used for the test case.
        description (str): A brief description of the test case.
    """
    ligand_name: str
    model: str
    description: str


tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define test cases
TEST_CASES = [
    StatsModeTestCase(
        "AF3",
        os.path.join(tests_dir, str(Path('./data/models/4dl8.cif'))),
        "Aluminum fluoride complex. mmCIF file"
    ),
    StatsModeTestCase(
        "AF3",
        os.path.join(tests_dir, str(Path('./data/models/4dl8.pdb'))),
        "Aluminum fluoride complex. pdb file"
    ),
    StatsModeTestCase(
        "AF3",
        '4dl8',
        "Aluminum fluoride complex. RCSB PDB ID"
    ),
]


@pytest.fixture(params=TEST_CASES)
def test_case(request: Any) -> StatsModeTestCase:
    """Fixture that provides the test case."""
    return request.param


@pytest.fixture
def temp_dir(test_case: StatsModeTestCase, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a unique temporary directory for each test case."""
    return tmp_path_factory.mktemp(f"data_{test_case.ligand_name}_{os.urandom(8).hex()}")


@pytest.fixture
def cli_output(temp_dir: Path, test_case: StatsModeTestCase) -> List:
    """Fixture that runs the CLI command and returns the output data."""
    model = test_case.model
    ligand_name = test_case.ligand_name
    test_args = [
        'metalCoord', 'stats', 
        '-l', ligand_name, 
        '-p', model, 
        '-o', os.path.join(temp_dir, f'{ligand_name}.json')
    ]
    
    # Run CLI command
    result = subprocess.run(
        test_args, capture_output=True, text=True, check=True
    )

    output_path = os.path.join(temp_dir, f'{ligand_name}.json')
    status_path = os.path.join(temp_dir, f'{ligand_name}.json.status.json')

    # Verify CLI execution
    assert result.returncode == 0, f"CLI stats command failed for {ligand_name}"
    assert Path(output_path).exists(), f"Output file not found for {ligand_name}"
    assert Path(status_path).exists(), f"Status file not found for {ligand_name}"

    # Check status file
    status = json.loads(Path(status_path).read_text(encoding='utf-8'))
    assert status["status"] == "Success", f"Status file does not contain success status for {ligand_name}"

    # Read and return the output data
    output_data = json.loads(Path(output_path).read_text(encoding='utf-8'))
    return output_data


@pytest.fixture
def validated_structure(cli_output: List) -> List:
    """Fixture that validates basic structure and returns validated data."""
    assert isinstance(cli_output, list), "Result should be a list"
    assert len(cli_output) > 0, "Result should not be empty"
    
    for entry in cli_output:
        required_fields = ["chain", "residue", "sequence", "metal", "metalElement", "ligands"]
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"
        
        assert isinstance(entry["ligands"], list), "Ligands should be a list"
        assert len(entry["ligands"]) > 0, "Ligands list should not be empty"
    
    return cli_output


@pytest.fixture
def validated_ligands(validated_structure: List) -> List:
    """Fixture that validates ligand structure and returns validated data."""
    for entry in validated_structure:
        for ligand in entry["ligands"]:
            required_fields = ["class", "procrustes", "coordination", "count", "description"]
            for field in required_fields:
                assert field in ligand, f"Missing required field in ligand: {field}"
            
            assert isinstance(ligand["class"], str), "Ligand class should be a string"
            assert isinstance(ligand["procrustes"], float), "Procrustes should be a float"
            assert isinstance(ligand["coordination"], int), "Coordination should be an integer"
            assert isinstance(ligand["count"], int), "Count should be an integer"
            assert isinstance(ligand["description"], str), "Description should be a string"
    
    return validated_structure


def test_main_func_stats_af3_cif_with_validation(validated_ligands: List, test_case: StatsModeTestCase):
    """Main test function that uses validated data."""
    # Test geometry classes
    expected_classes = {
        "octahedral",
        "trigonal-prism",
        "bicapped-square-planar",
        "sandwich_4_2",
        "sandwich_4h_2",
        "hexagonal-planar",
        "sandwich_5_1"
    }
    
    found_classes = set()
    for entry in validated_ligands:
        for ligand in entry["ligands"]:
            found_classes.add(ligand["class"])
    
    assert found_classes.issubset(expected_classes), \
        f"Unexpected geometry classes found for {test_case.ligand_name}: {found_classes - expected_classes}"

    # Test coordination numbers
    for entry in validated_ligands:
        for ligand in entry["ligands"]:
            assert 2 <= ligand["coordination"] <= 8, \
                f"Coordination number {ligand['coordination']} outside expected range (2-8) for {test_case.ligand_name}"

    # Test aluminum-specific properties
    al_entries = [entry for entry in validated_ligands if entry["metalElement"] == "Al"]
    for entry in al_entries:
        for ligand in entry["ligands"]:
            assert ligand["coordination"] == 6, f"Aluminum should have coordination number 6 for {test_case.ligand_name}"



def test_specific_af3_properties(validated_ligands: List, test_case: StatsModeTestCase):
    """Test specific properties expected for AF3 ligand."""
    for entry in validated_ligands:
        assert entry["residue"] == "AF3", f"Expected AF3 residue for {test_case.ligand_name}"
        
        # Test for expected ligand properties
        for ligand in entry["ligands"]:
            if ligand["class"] == "octahedral":
                assert ligand["procrustes"] < 0.1, \
                    f"Octahedral geometry should have low Procrustes value for {test_case.ligand_name}"
                if ligand["count"] > 0:  # If count is provided
                    assert ligand["count"] > 10, \
                        f"Expected multiple instances of octahedral geometry for {test_case.ligand_name}"
