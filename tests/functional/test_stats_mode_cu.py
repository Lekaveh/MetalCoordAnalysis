import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, NamedTuple
import pytest


class StatsModeTestCase(NamedTuple):
    """
    A test case class for statistical mode analysis of copper coordination in ligands.

    Attributes:
        ligand_name (str): The name of the ligand being analyzed.
        model (str): The model used for the analysis.
        parameters (Dict[str, str]): A dictionary of parameters used in the analysis.
        output_suffix (str): The suffix to be added to the output files.
        reference_file (str): The path to the reference file used for comparison.
        description (str): A brief description of the test case.
    """
    ligand_name: str
    model: str
    parameters: Dict[str, str]
    output_suffix: str
    reference_file: str
    description: str

def compare_specific_fields(data1: List[Dict], data2: List[Dict]) -> List[str]:
    """
    Compare specific fields in metal coordination data.
    Returns list of differences.
    """
    differences = []
    
    # Top level fields to check
    top_fields = ["chain", "residue", "sequence", "metal", "metalElement", "icode", "altloc"]
    # First ligand fields to check
    ligand_fields = ["class", "procrustes", "coordination", "count", "description"]
    
    # Compare first entry only
    if not data1 or not data2:
        return ["Empty data"]
    
    entry1 = data1[0]
    entry2 = data2[0]
    
    # Compare top level fields
    for field in top_fields:
        if field not in entry1 or field not in entry2:
            differences.append(f"Field '{field}' missing in one of the structures")
            continue
        
        val1 = entry1[field]
        val2 = entry2[field]
        
        if isinstance(val1, float) and isinstance(val2, float):
            if abs(val1 - val2) > 0.01:
                differences.append(f"Field '{field}' values differ: {val1} vs {val2}")
        elif val1 != val2:
            differences.append(f"Field '{field}' values differ: {val1} vs {val2}")
    
    # Compare first ligand fields
    if "ligands" not in entry1 or "ligands" not in entry2:
        differences.append("'ligands' field missing")
        return differences
    
    if not entry1["ligands"] or not entry2["ligands"]:
        differences.append("Empty ligands list")
        return differences
    
    ligand1 = entry1["ligands"][0]
    ligand2 = entry2["ligands"][0]
    
    for field in ligand_fields:
        if field not in ligand1 or field not in ligand2:
            differences.append(f"Ligand field '{field}' missing in one of the structures")
            continue
        
        val1 = ligand1[field]
        val2 = ligand2[field]
        
        if isinstance(val1, float) and isinstance(val2, float):
            if abs(val1 - val2) > 0.01:
                differences.append(f"Ligand field '{field}' values differ: {val1} vs {val2}")
        elif val1 != val2:
            differences.append(f"Ligand field '{field}' values differ: {val1} vs {val2}")
    
    return differences

# Test cases
TEST_CASES = [
    StatsModeTestCase(
        ligand_name="CU",
        model=str(Path("./tests/data/models/3kw8.cif")),
        parameters={},
        output_suffix="",
        reference_file=str(Path("./tests/data/results/3kw8_CU_mc.json")),
        description="Sodium coordination"
    ),
    StatsModeTestCase(
        ligand_name="CU",
        model=str(Path("./tests/data/models/3kw8.cif")),
        parameters={"-d": "0.4"},
        output_suffix="-d0p4",
        reference_file=str(Path("./tests/data/results/3kw8_CU_mc-d0p4.json")),
        description="Copper coordination with distance cutoff"
    ),
    StatsModeTestCase(
        ligand_name="CU",
        model=str(Path("./tests/data/models/3kw8.cif")),
        parameters={"-c": "3"},
        output_suffix="-c3",
        reference_file=str(Path("./tests/data/results/3kw8_CU_mc-c3.json")),
        description="Copper coordination with coordination number 3"
    )
]

@pytest.fixture(params=TEST_CASES)
def test_case(request: Any) -> StatsModeTestCase:
    """Fixture that provides the test case."""
    return request.param

@pytest.fixture
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path_factory.mktemp("metal_coordination_tests")

def test_metal_coordination_specific_fields(test_case: StatsModeTestCase, temp_dir: Path):
    """Test specific fields of metal coordination output."""
    # Build command
    output_file = f"{test_case.ligand_name}_mc{test_case.output_suffix}.json"
    output_path = os.path.join(temp_dir, output_file)
    
    cmd = [
        'metalCoord', 'stats',
        '-l', test_case.ligand_name,
        '-p', test_case.model,
        '-o', output_path
    ]
    
    # Add additional parameters
    for param, value in test_case.parameters.items():
        cmd.extend([param, value])
    
    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with: {result.stderr}"
    
    # Verify output file exists
    assert Path(output_path).exists(), f"Output file not created: {output_path}"
    
    # Load output and reference
    with open(output_path, 'r') as f:
        output_data = json.load(f)
    with open(test_case.reference_file, 'r') as f:
        reference_data = json.load(f)
    
    # Compare specific fields
    differences = compare_specific_fields(output_data, reference_data)
    
    # Report differences if any
    if differences:
        print(f"\nDifferences found for {test_case.description}:")
        for diff in differences:
            print(f"  {diff}")
        pytest.fail(f"Output differs from reference for {test_case.ligand_name}")

