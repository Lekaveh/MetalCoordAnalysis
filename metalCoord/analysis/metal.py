import json
import sys
from pathlib import Path


from metalCoord.analysis.models import MetalPairStats, Atom
from metalCoord.analysis.structures import MetalBondRegistry


class MetalPairStatsService:
    """
    A singleton class to handle statistics related to metal-metal distances.

    This class provides methods to retrieve mean distances and standard deviations
    between different metals based on preloaded data.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super(MetalPairStatsService, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def __init__(self):
        """
        Initialize the MetalMetalStats class by loading the metal-metal distances data.
        """
        if not hasattr(self, "data"):  # Prevent reinitialization
            data_path = (
                Path(sys.modules["metalCoord"].__file__).parent
                / "data/metal-metal distances.json"
            )
            with open(data_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
                self.data = self._process_data(raw_data)

    def _process_data(self, raw_data):
        """
        Process raw JSON data into a nested dictionary for easier access.

        Args:
            raw_data (list): A list of dictionaries containing metal-metal distance data.

        Returns:
            dict: A nested dictionary where the first key is metal1, the second key is metal2,
                  and the value is a tuple of (mean, std).
        """
        processed_data = {}
        for entry in raw_data:
            metal1, metal2 = entry["metal1"], entry["metal2"]
            mean, std = entry["mean"], entry["std"]
            if metal1 not in processed_data:
                processed_data[metal1] = {}
            processed_data[metal1][metal2] = (mean, std)
            # Ensure symmetry in the data
            if metal2 not in processed_data:
                processed_data[metal2] = {}
            processed_data[metal2][metal1] = (mean, std)
        return processed_data

    def get_distances_for_metal(self, metal_name):
        """
        Get all mean distances and standard deviations for a given metal with other metals.

        Args:
            metal_name (str): The name of the metal.

        Returns:
            dict: A dictionary where keys are other metal names and values are tuples of (mean, std).
        """
        if metal_name not in self.data:
            raise ValueError(f"No data available for metal: {metal_name}")
        return self.data[metal_name]

    def get_distance_between_metals(self, metal1, metal2):
        """
        Get the mean distance and standard deviation between two metals.

        Args:
            metal1 (str): The name of the first metal.
            metal2 (str): The name of the second metal.

        Returns:
            tuple: A tuple containing (mean, std) of the distance between the two metals.
        """
        if metal1 not in self.data or metal2 not in self.data[metal1]:
            raise ValueError(f"No data available for metals: {metal1} and {metal2}")
        return self.data[metal1][metal2]

    def has_distance_data(self, metal1, metal2):
        """
        Check if distance data exists between two metals.

        Args:
        metal1 (str): The name of the first metal.
        metal2 (str): The name of the second metal.

        Returns:
        bool: True if distance data exists, False otherwise.
        """
        return metal1 in self.data and metal2 in self.data[metal1]

    def has_distance_data_for_metal(self, metal):
        """
        Check if distance data exists for a given metal.

        Args:
        metal (str): The name of the metal.

        Returns:
        bool: True if distance data exists, False otherwise.
        """
        return metal in self.data

    def get_metal_pair_stats(self, metal_pairs: MetalBondRegistry):
        """
        Get statistics for a given set of metal pairs.

        Args:
            metal_pairs (MetalBondRegistry): A MetalBondRegistry object containing metal pairs.

        Returns:
            dict: A dictionary with metal pairs as keys and their statistics as values.
        """
        stats = []
        for bond in metal_pairs.get_bonds():
            if self.has_distance_data(bond.metal1.element, bond.metal2.element):
                stats.append(
                    MetalPairStats(
                        Atom(bond.metal1),
                        Atom(bond.metal2),
                        *self.get_distance_between_metals(
                            bond.metal1.element, bond.metal2.element
                        ),
                    )
                )
        return stats
