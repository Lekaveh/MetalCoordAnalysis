class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.distance_threshold = 0.2
            cls._instance.procrustes_threshold = 0.3
        return cls._instance
    
    def scale(self):
        return self.distance_threshold  + 1
    
    def procrustes_thr(self):
        return self.procrustes_threshold