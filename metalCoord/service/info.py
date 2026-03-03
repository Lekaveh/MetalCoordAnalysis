import json
import os
from pathlib import Path
import sys
from html import escape
import gemmi
from metalCoord.analysis.classes import idealClasses
from metalCoord.analysis.data import DB
from metalCoord.logging import Logger


d = os.path.dirname(sys.modules["metalCoord"].__file__)
mons = json.load(open(os.path.join(d, "data/mons.json"), encoding="utf-8"))


def get_metal_ligand_list() -> list:
    """
    Retrieves the list of metals and their corresponding ligands.

    Returns:
    list: A list of metals and their corresponding ligands.

    """
    return mons.keys()


def get_pdbs_list(ligand: str) -> list:
    """
    Retrieves the PDB files containing the given ligand.

    Parameters:
    ligand (str): The name of the ligand.

    Returns:
    list: A list of PDB files containing the ligand.

    """
    if ligand in mons:
        return {
            ligand: sorted(
                mons[ligand], key=lambda x: (not x[2], x[1] if x[1] else 10000)
            )
        }
    return {ligand: []}


def process_pdbs_list(ligand: str, output: str) -> list:
    """
    Retrieves the PDB files containing the given ligand and optionally writes them to a JSON file.

    output (str): The file path to write the list of PDB files. If not provided, the list will be printed.

    """
    pdbs = get_pdbs_list(ligand)
    if output:
        directory = os.path.dirname(output)
        Path(directory).mkdir(exist_ok=True, parents=True)
        with open(output, "w", encoding="utf-8") as json_file:
            json.dump(pdbs, json_file, indent=4, separators=(",", ": "))
            Logger().info(f"List of pdbs for {ligand} written to {output}")
    else:
        print(pdbs)


def get_coordinations(
    coordination_num: int = None, metal: str = None, cod: bool = False
) -> list:
    """
    Retrieve coordination information based on the provided parameters.
    Args:
        coordination_num (int, optional): The coordination number to filter by. Defaults to None.
        metal (str, optional): The metal element to filter by. Defaults to None.
        cod (bool, optional): Whether to include the COD IDs in the result. Defaults to False.
    Returns:
        list: A list of coordination information based on the provided parameters.
    Raises:
        ValueError: If the provided metal is not a valid metal element.
    Notes:
        - If only `coordination_num` is provided, returns ideal classes by coordination number.
        - If both `metal` and `coordination_num` are provided, returns frequency data for the metal and coordination number.
        - If only `metal` is provided, returns frequency data for the metal across all coordination numbers.
        - If neither `coordination_num` nor `metal` is provided, returns all ideal classes.
    """

    if metal:
        if not gemmi.Element(metal).is_metal:
            raise ValueError(f"{metal} is not a metal element.")
        metal = metal.lower().capitalize()
    if coordination_num and not metal:
        return DB.get_frequency_coordination(coordination_num, cod=cod)

    if metal and coordination_num:
        return DB.get_frequency_metal_coordination(metal, coordination_num, cod=cod)

    if metal and not coordination_num:
        return DB.get_frequency_metal(metal, cod=cod)

    return DB.get_frequency(cod=cod)


def process_coordinations(
    coordination_num: int = None,
    metal: str = None,
    output: str = None,
    cod: bool = False,
) -> None:
    """
    Retrieve coordination information based on the provided parameters and optionally write it to a JSON file.
    Args:
        coordination_num (int, optional): The coordination number to filter by. Defaults to None.
        metal (str, optional): The metal element to filter by. Defaults to None.
        output (str, optional): The file path to write the coordination information. Defaults to None.
        cod (bool, optional): Whether to include the COD IDs in the result. Defaults to False.
    Raises:
        ValueError: If the provided metal is not a valid metal element.
    Notes:
        - If only `coordination_num` is provided, writes ideal classes by coordination number to a JSON file.
        - If both `metal` and `coordination_num` are provided, writes frequency data for the metal and coordination number to a JSON file.
        - If only `metal` is provided, writes frequency data for the metal across all coordination numbers to a JSON file.
        - If neither `coordination_num` nor `metal` is provided, writes all ideal classes to a JSON file.
    """
    coordinations = get_coordinations(coordination_num, metal, cod=cod)
    if output:
        directory = os.path.dirname(output)
        Path(directory).mkdir(exist_ok=True, parents=True)
        with open(output, "w", encoding="utf-8") as json_file:
            json.dump(coordinations, json_file, indent=4, separators=(",", ": "))
            Logger().info(f"Coordinations info written to {output}")
    else:
        print(coordinations)


def get_ideal_classes_table() -> list:
    """
    Build a table for all ideal classes with class code and atom coordinates.

    Returns:
        list: List of records, one per ideal class.
    """
    rows = []
    for class_name in idealClasses.get_ideal_classes():
        atom_coordinates = []
        for index, coord in enumerate(idealClasses.get_coordinates(class_name), start=1):
            atom_coordinates.append(
                {
                    "atom_index": index,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                }
            )
        rows.append(
            {
                "class_name": class_name,
                "class_code": idealClasses.get_class_code(class_name),
                "atom_coordinates": atom_coordinates,
            }
        )
    return rows


def _render_table_html(table_rows: list) -> str:
    row_lines = []
    for row in table_rows:
        coordinates_json = json.dumps(row["atom_coordinates"], indent=2)
        row_lines.append(
            "        <tr>"
            f"<td>{escape(row['class_name'])}</td>"
            f"<td>{escape(row['class_code'])}</td>"
            f"<td><pre>{escape(coordinates_json)}</pre></td>"
            "</tr>"
        )

    rows_html = "\n".join(row_lines)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <title>MetalCoord Ideal Classes Table</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 1.5rem; }\n"
        "    table { border-collapse: collapse; width: 100%; }\n"
        "    th, td { border: 1px solid #d9d9d9; padding: 0.5rem; text-align: left; vertical-align: top; }\n"
        "    th { background: #f6f6f6; }\n"
        "    pre { margin: 0; white-space: pre-wrap; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>MetalCoord Ideal Classes</h1>\n"
        "  <table>\n"
        "    <thead>\n"
        "      <tr><th>Class Name</th><th>Class Code</th><th>Atom Coordinates</th></tr>\n"
        "    </thead>\n"
        "    <tbody>\n"
        f"{rows_html}\n"
        "    </tbody>\n"
        "  </table>\n"
        "</body>\n"
        "</html>\n"
    )


def process_table(output_folder: str = ".", output_format: str = "json") -> list:
    """
    Write a table of ideal classes to JSON, HTML, or both.

    Args:
        output_folder (str): Destination folder for output files.
        output_format (str): json, html, or both.

    Returns:
        list: Paths to files written.
    """
    output_format = (output_format or "json").lower()
    if output_format not in {"json", "html", "both"}:
        raise ValueError(f"Unsupported format: {output_format}")

    output_dir = os.path.abspath(output_folder or ".")
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    table_rows = get_ideal_classes_table()
    output_paths = []

    if output_format in {"json", "both"}:
        json_output = os.path.join(output_dir, "metalcoord_table.json")
        with open(json_output, "w", encoding="utf-8") as json_file:
            json.dump(table_rows, json_file, indent=4, separators=(",", ": "))
        output_paths.append(json_output)
        Logger().info(f"Class table (JSON) written to {json_output}")

    if output_format in {"html", "both"}:
        html_output = os.path.join(output_dir, "metalcoord_table.html")
        with open(html_output, "w", encoding="utf-8") as html_file:
            html_file.write(_render_table_html(table_rows))
        output_paths.append(html_output)
        Logger().info(f"Class table (HTML) written to {html_output}")

    return output_paths
