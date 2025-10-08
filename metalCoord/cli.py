"""Command-line interface utilities for the MetalCoord package."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, Sequence, TypeVar

import metalCoord
from metalCoord.config import Config


class Range:
    """Inclusive numeric range used for ``argparse`` choices."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end

    def __eq__(self, other: object) -> bool:
        try:
            value = float(other)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self.start <= value <= self.end

    def __repr__(self) -> str:  # pragma: no cover - representational helper
        return f"range({self.start}, {self.end})"

    __str__ = __repr__



def _check_positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return number


def _check_positive_more_than_two(value: str) -> int:
    number = int(value)
    if number <= 1:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value. It should be more than 1"
        )
    return number


@dataclass
class AnalysisCommandArgs:
    """Options shared by the ``update`` and ``stats`` commands."""

    output: str
    dist: float
    threshold: float
    min_size: int
    max_size: int
    ideal_angles: bool
    simple: bool
    save: bool
    use_pdb: bool
    coordination: int
    clazz: Optional[str]

    def validate(self) -> None:
        if self.min_size > self.max_size:
            raise ValueError(
                "Minimum sample size must be less or equal than maximum sample size."
            )

    def apply_to_config(self, config: Config) -> None:
        config.distance_threshold = self.dist
        config.procrustes_threshold = self.threshold
        config.min_sample_size = self.min_size
        config.max_sample_size = self.max_size
        config.simple = self.simple
        config.save = self.save
        config.ideal_angles = self.ideal_angles
        config.use_pdb = self.use_pdb
        config.max_coordination_number = self.coordination
        config.output_folder = os.path.abspath(os.path.dirname(self.output))
        config.output_file = os.path.basename(self.output)
        setattr(config, "clazz", self.clazz)


@dataclass
class UpdateArgs(AnalysisCommandArgs):
    input: str
    pdb: Optional[str]
    cif: bool

    def apply_to_config(self, config: Config) -> None:  # type: ignore[override]
        super().apply_to_config(config)


@dataclass
class StatsArgs(AnalysisCommandArgs):
    ligand: str
    pdb: str
    metal_distance: float

    def apply_to_config(self, config: Config) -> None:  # type: ignore[override]
        super().apply_to_config(config)
        config.metal_distance_threshold = self.metal_distance


@dataclass
class CoordinationArgs:
    number: Optional[int]
    metal: Optional[str]
    output: Optional[str]
    cod: bool


@dataclass
class PdbArgs:
    ligand: str
    output: Optional[str]


CommandArgs = TypeVar(
    "CommandArgs", UpdateArgs, StatsArgs, CoordinationArgs, PdbArgs
)


@dataclass
class CommandResult(Generic[CommandArgs]):
    """Dataclass representing the parsed CLI invocation."""

    command: Literal["update", "stats", "coord", "pdb"]
    args: CommandArgs
    no_progress: bool

    @property
    def is_analysis(self) -> bool:
        return isinstance(self.args, AnalysisCommandArgs)


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--dist",
        type=float,
        default=0.5,
        metavar="<DISTANCE THRESHOLD>",
        choices=[Range(0, 1)],
        help="Distance threshold.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        metavar="<PROCRUSTES DISTANCE THRESHOLD>",
        choices=[Range(0, 1)],
        help="Procrustes distance threshold.",
    )
    parser.add_argument(
        "-m",
        "--min_size",
        type=_check_positive,
        default=30,
        metavar="<MINIMUM SAMPLE SIZE>",
        help="Minimum sample size for statistics.",
    )
    parser.add_argument(
        "-x",
        "--max_size",
        type=_check_positive,
        default=2000,
        metavar="<MAXIMUM SAMPLE SIZE>",
        help="Maximum sample size for statistics.",
    )
    parser.add_argument(
        "--ideal-angles",
        action="store_true",
        help="Provide only ideal angles",
    )
    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="Simple distance based filtering",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save COD files used in statistics",
    )
    parser.add_argument(
        "--use-pdb",
        action="store_true",
        help="Use COD structures based on pdb coordinates",
    )
    parser.add_argument(
        "-c",
        "--coordination",
        type=_check_positive_more_than_two,
        default=1000,
        metavar="<MAXIMUM COORDINATION NUMBER>",
        help="Maximum coordination number.",
    )
    parser.add_argument(
        "--cl",
        type=str,
        metavar="<CLASS>",
        help="Predefined class/coordination",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metalCoord", description="MetalCoord: Metal coordination analysis."
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + metalCoord.__version__,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not show progress bar.",
    )

    subparsers = parser.add_subparsers(dest="command")

    update_parser = subparsers.add_parser("update", help="Update a cif file.")
    update_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        metavar="<INPUT CIF FILE>",
        help="CIF file.",
    )
    update_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="<OUTPUT CIF FILE>",
        help="Output cif file.",
    )
    update_parser.add_argument(
        "-p",
        "--pdb",
        type=str,
        metavar="<PDB CODE|PDB FILE>",
        help="PDB code or pdb file.",
    )
    _add_analysis_arguments(update_parser)
    update_parser.add_argument(
        "--cif",
        action="store_true",
        help="Read coordinates from mmCIF file",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="Distance and angle statistics."
    )
    stats_parser.add_argument(
        "-l",
        "--ligand",
        type=str,
        required=True,
        metavar="<LIGAND CODE>",
        help="Ligand code.",
    )
    stats_parser.add_argument(
        "-p",
        "--pdb",
        type=str,
        required=True,
        metavar="<PDB CODE|PDB FILE>",
        help="PDB code or pdb file.",
    )
    stats_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="<OUTPUT JSON FILE>",
        help="Output json file.",
    )
    _add_analysis_arguments(stats_parser)
    stats_parser.add_argument(
        "--metal_distance",
        type=float,
        default=0.3,
        metavar="<METAL DISTANCE THRESHOLD>",
        choices=[Range(0, 1)],
        help="Metal Metal distance threshold.",
    )

    coordination_parser = subparsers.add_parser(
        "coord", help="List of coordinations."
    )
    coordination_parser.add_argument(
        "-n",
        "--number",
        type=int,
        metavar="<COORDINATION NUMBER>",
        help="Coordination number.",
    )
    coordination_parser.add_argument("-m", "--metal", type=str, help="Metal code.")
    coordination_parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="<OUTPUT JSON FILE>",
        help="Output json file.",
    )
    coordination_parser.add_argument(
        "--cod",
        action="store_true",
        help="Include IDs of the COD structures.",
    )

    pdb_parser = subparsers.add_parser(
        "pdb", help="Get list of PDBs containing the ligand."
    )
    pdb_parser.add_argument(
        "-l",
        "--ligand",
        type=str,
        required=True,
        metavar="<LIGAND CODE>",
        help="Ligand code.",
    )
    pdb_parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="<OUTPUT JSON FILE>",
        help="Output json file.",
    )
    return parser


def _namespace_to_update(namespace: argparse.Namespace) -> UpdateArgs:
    return UpdateArgs(
        input=namespace.input,
        pdb=namespace.pdb,
        cif=bool(getattr(namespace, "cif", False)),
        output=namespace.output,
        dist=namespace.dist,
        threshold=namespace.threshold,
        min_size=namespace.min_size,
        max_size=namespace.max_size,
        ideal_angles=bool(getattr(namespace, "ideal_angles", False)),
        simple=bool(getattr(namespace, "simple", False)),
        save=bool(getattr(namespace, "save", False)),
        use_pdb=bool(getattr(namespace, "use_pdb", False)),
        coordination=namespace.coordination,
        clazz=getattr(namespace, "cl", None),
    )


def _namespace_to_stats(namespace: argparse.Namespace) -> StatsArgs:
    return StatsArgs(
        ligand=namespace.ligand,
        pdb=namespace.pdb,
        metal_distance=namespace.metal_distance,
        output=namespace.output,
        dist=namespace.dist,
        threshold=namespace.threshold,
        min_size=namespace.min_size,
        max_size=namespace.max_size,
        ideal_angles=bool(getattr(namespace, "ideal_angles", False)),
        simple=bool(getattr(namespace, "simple", False)),
        save=bool(getattr(namespace, "save", False)),
        use_pdb=bool(getattr(namespace, "use_pdb", False)),
        coordination=namespace.coordination,
        clazz=getattr(namespace, "cl", None),
    )


def _namespace_to_coordination(namespace: argparse.Namespace) -> CoordinationArgs:
    return CoordinationArgs(
        number=namespace.number,
        metal=namespace.metal,
        output=namespace.output,
        cod=bool(getattr(namespace, "cod", False)),
    )


def _namespace_to_pdb(namespace: argparse.Namespace) -> PdbArgs:
    return PdbArgs(ligand=namespace.ligand, output=namespace.output)


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> CommandResult[Any]:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.command is None:
        parser.print_help()
        raise SystemExit(1)

    converters = {
        "update": _namespace_to_update,
        "stats": _namespace_to_stats,
        "coord": _namespace_to_coordination,
        "pdb": _namespace_to_pdb,
    }
    args = converters[namespace.command](namespace)

    if isinstance(args, AnalysisCommandArgs):
        args.validate()

    return CommandResult(
        command=namespace.command,
        args=args,
        no_progress=bool(getattr(namespace, "no_progress", False)),
    )
