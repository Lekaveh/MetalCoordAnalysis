# MetalCoord
# Metal Coordination Analysis Tool

## Overview

The Metal Coordination Analysis Tool is a Python application designed for analyzing metal coordination in biological macromolecules such as proteins and nucleic acids. This tool provides a set of functionalities to process molecular structures, identify metal coordination sites, and generate insightful visualizations.

## Features

- **Metal Coordination Analysis**: The application can automatically detect metal coordination sites within given molecular structures and provide distance and angle statistics for these metal structures.

- **Ligand Update**: Based coordination analysis update distances and angles of specified metal containing ligands.

## Installation

### Prerequisites

- Python 3.x
- gemmi>=0.6.2
- numpy>=1.20
- pandas>=2.0.0
- tensorflow>=2.9.1
- tqdm>=4.0.0
- scipy>=1.0.0

### Installation

pip install git+https://github.com/Lekaveh/MetalCoordAnalysis


### Usage
- **Metal Coordination Analysis**: 
    - metalCoord stats [-h] -l <LIGAND CODE> -p <PDB CODE|PDB FILE> -o <OUTPUT JSON FILE>
- **Ligand Update**: 
    - metalCoord update [-h] -i <INPUT CIF FILE> -o <OUTPUT CIF FILE> [-p <PDB CODE|PDB FILE>]