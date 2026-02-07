# MetalCoord
# Metal Coordination Analysis Tool

[Improving macromolecular structure refinement with metal-coordination restraints](https://doi.org/10.1107/S2059798324011458)

Kaveh H. Babai, Fei Long, Martin Malý, Keitaro Yamashitad and Garib N. Murshudov (2024)

Acta Crystallogr. D80, Part 12

## Overview

The Metal Coordination Analysis Tool is a Python application designed for analyzing metal coordination in biological macromolecules such as proteins and nucleic acids. This tool provides a set of functionalities to process molecular structures, identify metal coordination sites, and generate insightful visualizations.

## Features

- **Metal Coordination Analysis**: The application can automatically detect metal coordination sites within given molecular structures and provide distance and angle statistics for these metal structures.

- **Ligand Update**: Based coordination analysis update distances and angles of specified metal containing ligands.

- **Coordination List**: Provides a list of coordination numbers for the given parameters.

- **Pdb list**: Provide  list of pdb with their resolution for the specific ligand

## Installation

### Prerequisites

- Python>=3.9
- gemmi>=0.6.2
- numpy>=1.26
- pandas>=2.0.0
- tqdm>=4.0.0
- scipy>=1.0.0
- scikit-learn>=1.4.0

### Installation

pip install git+https://github.com/Lekaveh/MetalCoordAnalysis


### Usage
- **Metal Coordination Analysis**: 
    - metalCoord [--no-progress] stats -p <PDB_CODE|PDB_FILE> -o OUTPUT_JSON_FILE [-l <LIGAND_CODE> ] [-d <DISTANCE_THRESHOLD>] [-t <PROCRUSTES_DISTANCE_THRESHOLD>] [-m <MINIMUM_SAMPLE_SIZE>] [--ideal_angles] [-s] [--save] [--use-pdb] [-c <MAXIMUM_COORDINATION_NUMBER>] [--cl <PREDEFINED_CLASS>] [--debug] [--debug-level <summary|detailed|max>] [--debug-output <DEBUG_OUTPUT>]
        - -l, --ligand: Ligand code. If not specified, all metal-containing ligands will be analyzed.
        - -o, --output: Output JSON file.
        - -p, --pdb: PDB code or PDB file.
        - -d, --dist: Distance threshold (default: 0.5, range: 0-1).<br> *A threshold to* $d$ *to select atom is* $(r_1 + r_2)*(1 + d)$
        - -t, --threshold: Procrustes distance threshold (default: 0.3, range: 0-1).
        - -m, --min_size: Minimum sample size for statistics (default: 30).
        - -x, --max_size: Maximum sample size for statistics (default: 2000).
        - --ideal-angles: Provide only ideal angles.
        - -s, --simple: Simple distance-based filtering.
        - --save: Save COD files used in statistics.
        - --use-pdb Use COD structures based on pdb coordinates.
        - -c, --coordination: Defines maximum coordination number.
        - --cl Predefined class or  "most_common" for the most frequent coordination class
        - --metal_distance: Metal-metal distance threshold (default: 0.3, range: 0-1).
        - --debug: Write debug sidecar files (`.debug.json` and `.debug.md`).
        - --debug-level: Debug detail level (`summary`, `detailed`, or `max`; default: `detailed`).
        - --debug-output: Optional debug output override. For a single output, use file or directory. For multi-ligand stats, this must be a directory.
        - --no-progress: Do not show progress bars
- **Ligand Update**: 
    - metalCoord [--no-progress] update -i <INPUT_CIF_FILE> -o <OUTPUT_CIF_FILE> [-p <PDB_CODE|PDB_FILE>] [-d <DISTANCE_THRESHOLD>] [-t <PROCRUSTES_DISTANCE_THRESHOLD>] [-m <MINIMUM_SAMPLE_SIZE>] [--ideal_angles] [-s] [--save] [--use-pdb] [-c <MAXIMUM_COORDINATION_NUMBER>] [--cif] [--cl <PREDEFINED_CLASS>] [--debug] [--debug-level <summary|detailed|max>] [--debug-output <DEBUG_OUTPUT>]
        - -i, --input: CIF file.
        - -o, --output: Output CIF file.
        - -p, --pdb: PDB code or PDB file.
        - -d, --dist: Distance threshold (default: 0.5, range: 0-1).<br> *A threshold to* $d$ *to select atom is* $(r_1 + r_2)*(1 + d)$
        - -t, --threshold: Procrustes distance threshold (default: 0.3, range: 0-1).
        - -m, --min_size: Minimum sample size for statistics (default: 30).
        - -x, --max_size: Maximum sample size for statistics (default: 2000).
        - --ideal-angles: Provide only ideal angles.
        - -s, --simple: Simple distance-based filtering.
        - --use-pdb Use COD structures based on pdb coordinates.
        - -c, --coordination: Defines maximum coordination number.
        - --cif: Read coordinates from mmCIF file
        - --cl Predefined class or  "most_common" for the most frequent coordination class
        - --debug: Write debug sidecar files (`.debug.json` and `.debug.md`).
        - --debug-level: Debug detail level (`summary`, `detailed`, or `max`; default: `detailed`).
        - --debug-output: Optional debug output override (file or directory).
        - --no-progress: Do not show progress bars
- **Coordination List**: 
    - metalCoord [--no-progress] coord [-n <COORDINATION_NUMBER>] [-m <METAL_ELEMENT_NAME>] [-o <OUTPUT_JSON_FILE>] [--cod]
        - -n, --number: Coordination number.
        - -m, --metal: Metal element name.
        - -o, --output: Output JSON file
        - --cod: Include IDs of the COD structures
        - --no-progress: Do not show progress bars
- **List of PDBs**: 
    - metalCoord [--no-progress] pdb -l <LIGAND_CODE> [-o <OUTPUT_JSON_FILE>]
        - -l, --ligand: Ligand code.
        - -o, --output: Output JSON file
        - --no-progress: Do not show progress bars

### Debug mode

When `--debug` is enabled for `stats` or `update`, MetalCoord writes two sidecar files next to the main output by default:

- `<output>.debug.json`
- `<output>.debug.md`

The debug JSON contains:

- `domain_report`: chemistry-first stepwise narrative of the analysis process.
- `descriptor_info`: linear descriptor generation details (class code, ordering, index mapping, and descriptor string).
- `trace`: structured analysis trace for sites, candidates, and selected strategy.
- `logs`: captured runtime logs.

Debug level controls detail:

- `summary`: chosen descriptor/class per metal site.
- `detailed`: chosen + top candidate geometries (up to top 3 plus chosen if different).
- `max`: all candidates plus raw arrays where available.

`--debug-output` behavior:

- single-output `stats` or `update`: may be a file path or directory path.
- multi-ligand `stats` (no `-l`): must be a directory.

### Tutorial
For a step-by-step tutorial on how to use the Metal Coordination Analysis Tool, visit the [tutorial page](https://github.com/Lekaveh/MetalCoordAnalysis/blob/master/tutorial/tutorial.rst).

This ensures users have clear guidance and an easy way to access additional information through the tutorial.


