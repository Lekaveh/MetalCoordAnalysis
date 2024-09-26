# MetalCoord
# Metal Coordination Analysis Tool

## Overview

The Metal Coordination Analysis Tool is a Python application designed for analyzing metal coordination in biological macromolecules such as proteins and nucleic acids. This tool provides a set of functionalities to process molecular structures, identify metal coordination sites, and generate insightful visualizations.

## Features

- **Metal Coordination Analysis**: The application can automatically detect metal coordination sites within given molecular structures and provide distance and angle statistics for these metal structures.

- **Ligand Update**: Based coordination analysis update distances and angles of specified metal containing ligands.

- **Coordination List**: Provides a list of coordination numbers for the given parameters.

## Installation

### Prerequisites

- Python 3.x
- gemmi>=0.6.2
- numpy>=1.20
- pandas>=2.0.0
- tensorflow>=2.9.1
- tqdm>=4.0.0
- scipy>=1.0.0
- scikit-learn>=1.4.0

### Installation

pip install git+https://github.com/Lekaveh/MetalCoordAnalysis


### Usage
- **Metal Coordination Analysis**: 
    - metalCoord stats -l LIGAND_CODE -p <PDB_CODE|PDB_FILE> -o OUTPUT_JSON_FILE [-d <DISTANCE_THRESHOLD>] [-t <PROCRUSTES_DISTANCE_THRESHOLD>] [-m <MINIMUM_SAMPLE_SIZE>] [--ideal_angles] [-s] [--save]
        - -l, --ligand: Ligand code.
        - -o, --output: Output JSON file.
        - -p, --pdb: PDB code or PDB file.
        - -d, --dist: Distance threshold (default: 0.5, range: 0-1).
        - -t, --threshold: Procrustes distance threshold (default: 0.3, range: 0-1).
        - -m, --min_size: Minimum sample size for statistics (default: 30).
        - --ideal-angles: Provide only ideal angles.
        - -s, --simple: Simple distance-based filtering.
        - --save: Save COD files used in statistics.
        - --use-pdb Use COD structures based on pdb coordinates.
        - -c, --coordination: Defines maximum coordination number.
- **Ligand Update**: 
    - metalCoord update -i INPUT_CIF_FILE -o OUTPUT_CIF_FILE [-p <PDB_CODE|PDB_FILE>] [-d <DISTANCE_THRESHOLD>] [-t <PROCRUSTES_DISTANCE_THRESHOLD>] [-m <MINIMUM_SAMPLE_SIZE>] [--ideal_angles] [-s] [--save]
        - -i, --input: CIF file.
        - -o, --output: Output CIF file.
        - -p, --pdb: PDB code or PDB file.
        - -d, --dist: Distance threshold (default: 0.5, range: 0-1).
        - -t, --threshold: Procrustes distance threshold (default: 0.3, range: 0-1).
        - -m, --min_size: Minimum sample size for statistics (default: 30).
        - --ideal-angles: Provide only ideal angles.
        - -s, --simple: Simple distance-based filtering.
        - --save: Save COD files used in statistics.
        - --use-pdb Use COD structures based on pdb coordinates.
        - -c, --coordination: Defines maximum coordination number.
- **Coordination List**: 
    - metalCoord coord [-n <COORDINATION_NUMBER>]
        - -n, --number: Coordination number.

