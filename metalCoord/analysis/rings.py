import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AngleValue:
    angle: float
    std: float


@dataclass(frozen=True)
class RingAngleStatistics:
    metal1: AngleValue
    ligand1: AngleValue
    metal2: AngleValue
    ligand2: AngleValue


class RingAngleStatsService:
    """Load ring-angle statistics from packaged reference data."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        if data_path is None:
            data_path = (
                Path(sys.modules["metalCoord"].__file__).parent
                / "data/ring_angles.json"
            )
        with open(data_path, "r", encoding="utf-8") as file:
            self._data = json.load(file)

    @staticmethod
    def _angle_value(value: dict) -> AngleValue:
        return AngleValue(angle=value["angle"], std=value["std"])

    def find(
        self,
        elements: tuple[str, str, str, str],
        coordinations: tuple[int, int],
        classes: tuple[str, str],
    ) -> Optional[RingAngleStatistics]:
        """Return statistics for metal-ligand-metal-ligand ring description."""
        normalized_elements = tuple(element.upper() for element in elements)
        for entry in self._data:
            entry_elements = tuple(
                element.upper() for element in entry["elements"]
            )
            if (
                normalized_elements != entry_elements
                or tuple(coordinations) != tuple(entry["coordinations"])
                or tuple(classes) != tuple(entry["classes"])
            ):
                continue

            angles = entry["angles"]
            return RingAngleStatistics(
                metal1=self._angle_value(angles["metal1"]),
                ligand1=self._angle_value(angles["ligand1"]),
                metal2=self._angle_value(angles["metal2"]),
                ligand2=self._angle_value(angles["ligand2"]),
            )
        return None


RING_ANGLE_STATS = RingAngleStatsService()
