

class Config:
    """
    A class representing the configuration settings for MetalCoordAnalysis.

    Attributes:
        distance_threshold (float): The distance threshold for MetalCoordAnalysis.
        procrustes_threshold (float): The procrustes threshold for MetalCoordAnalysis.
        min_sample_size (int): The minimum sample size for MetalCoordAnalysis.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.distance_threshold = 0.2
            cls._instance.procrustes_threshold = 0.3
            cls._instance.min_sample_size = 30
        return cls._instance


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
