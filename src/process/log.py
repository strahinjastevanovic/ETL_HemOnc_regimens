import os 
import logging
class Logger:
    def __init__(self, log_dir, filename="PRE.processing.log", level="DEBUG"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler = logging.FileHandler(self.log_path, mode="w")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg): self.logger.info(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)
