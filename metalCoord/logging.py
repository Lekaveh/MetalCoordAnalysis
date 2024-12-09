import logging
import sys

class Logger:
    """Singleton Logger class for unified logging across an application."""
    _instances = {}

    def __new__(cls, name="MetalCoord", level=logging.DEBUG):
        """Override __new__ to implement the singleton pattern."""
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.__setup_logger(name, level)
            cls._instances[name] = instance
        return cls._instances[name]

    def __setup_logger(self, name, level):
        """Setup logger with specified name and level."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.__enabled = False
        self.__progress_bars = False

    @property
    def enabled(self):
        """Check if logger is enabled."""
        return self.__enabled

    @property
    def disabled(self):
        """Check if logger is disabled."""
        return not self.__enabled
    
    @property
    def progress_bars(self):
        """Check if progress bars are enabled."""
        return self.__progress_bars
    

    def add_handler(self, enable = True, progress_bars=True):
        """Add a stream handler to the logger with a standard format."""
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.__enabled = enable
        self.__progress_bars = progress_bars

    def debug(self, message):
        """Log a debug message."""
        if self.__enabled:
            self.logger.debug(message)

    def info(self, message):
        """Log an info message."""
        if self.__enabled:
            self.logger.info(message)

    def warning(self, message):
        """Log a warning message."""
        if self.__enabled:
            self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        if self.__enabled:
            self.logger.error(message)

    def critical(self, message):
        """Log a critical message."""
        if self.__enabled:
            self.logger.critical(message)