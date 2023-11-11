
import logging

class Logger:
    _instance = {}

    def __new__(cls, name = "MetalCoord", level=logging.DEBUG):
        if name not in cls._instance:
            cls._instance[name] = super().__new__(cls)
            cls._instance[name].logger = logging.getLogger(name)
            cls._instance[name].logger.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            cls._instance[name].logger.addHandler(ch)
        return cls._instance[name]

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
