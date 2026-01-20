
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Configuration settings for MetalCoordAnalysis.

    Attributes:
        distance_threshold (float): The distance threshold for MetalCoordAnalysis.
        procrustes_threshold (float): The procrustes threshold for MetalCoordAnalysis.
        min_sample_size (int): The minimum sample size for MetalCoordAnalysis.
    """

    distance_threshold: float = 0.2
    procrustes_threshold: float = 0.3
    metal_distance_threshold: float = 0.3
    min_sample_size: int = 30
    max_sample_size: Optional[int] = None
    simple: bool = False
    save: bool = False
    ideal_angles: bool = False
    use_pdb: bool = False
    max_coordination_number: Optional[int] = None
    output_folder: str = ""
    output_file: str = ""

    def scale(self) -> float:
        """
        Returns the scaled value of the distance threshold.

        The scaled value is obtained by adding 1 to the distance threshold.

        Returns:
            float: The scaled value of the distance threshold.
        """
        return self.distance_threshold + 1

    def procrustes_thr(self) -> float:
        """
        Returns the Procrustes threshold value.

        Returns:
            float: The Procrustes threshold value.
        """
        return self.procrustes_threshold
    
    def metal_scale(self) -> float:
        """
        Returns the scaled value of the metal distance threshold.

        The scaled value is obtained by adding 1 to the metal distance threshold.

        Returns:
            float: The scaled value of the metal distance threshold.
        """
        return self.metal_distance_threshold + 1
