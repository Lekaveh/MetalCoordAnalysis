

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
            cls._instance.__initialized = False
        return cls._instance


    def __init__(self):
        if self.__initialized:
            return
        self.distance_threshold = 0.2
        self.procrustes_threshold = 0.3
        self.min_sample_size = 30
        self.simple = False
        self.save = False
        self.ideal_angles = False
        self.use_pdb = False
        self.max_coordination_number = None
        self.output_folder = ""
        self.__initialized = True

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
