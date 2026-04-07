import os
import logging
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

class AppLogger:
    def __init__(self, name: str = "app_logger", log_dir: str = "logs", level=logging.INFO):
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=10)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, message: str, **context):
        self.logger.info(f"{message} | context: {context}")

    def warning(self, message: str, **context):
        self.logger.warning(f"{message} | context: {context}")

    def error(self, message: str, exc: Exception = None, **context):
        if exc:
            exc_info = "".join(traceback.format_exception(None, exc, exc.__traceback__))
            self.logger.error(f"{message} | Exception: {exc_info} | context: {context}")
        else:
            self.logger.error(f"{message} | context: {context}")

    def debug(self, message: str, **context):
        self.logger.debug(f"{message} | context: {context}")