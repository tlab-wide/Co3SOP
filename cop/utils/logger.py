import os
import loguru
import datetime

class Logger(object):
    def __init__(self, name) -> None:
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        self.logger = loguru.logger
        self.logger.add(
            f"./logs/{name}.log",
            format="{message}",
            rotation="30 day",
            encoding="utf-8"
        )
    
    def _log(self, level: str, message: str):
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        elif level == "WARNING":
            self.logger.warning(message)

    def error(self, message: str):
        self._log("ERROR", message)

    def debug(self, message: str):
        self._log("DEBUG", message)

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

logger = Logger("train")