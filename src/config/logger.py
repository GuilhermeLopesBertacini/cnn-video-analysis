"""Module describing logger_class for logging - not exeutable"""
import os
import time
import logging
from src.config.settings import LOGS_DIR

class LoggerClass:
    """Job: abstract all log generation/configuration"""

    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(LOGS_DIR, 
                                              time.strftime("%Y.%m.%d_%H:%M:%S_") + 'default.log'),
                                              format='%(asctime)s %(levelname)s: %(message)s')
    
    @classmethod
    def configure(cls, file_name: str, debug: bool = False):
        """Configures the logger with the file name and debug mode"""

        cls.script_name = time.strftime("%Y.%m.%d_%H:%M:%S_") + file_name
        logging.basicConfig(filename=os.path.join(LOGS_DIR, f'{cls.script_name}.log'),
                            format='%(asctime)s %(levelname)s: %(message)s')
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        # Aumentar o nível de log do APScheduler para WARNING (ou ERROR)
        logging.getLogger('apscheduler').setLevel(logging.WARNING)

        # Aumentar o nível de log do HTTPX para WARNING (ou ERROR)
        logging.getLogger('httpx').setLevel(logging.WARNING)

    @classmethod
    def get_log_path(cls):
        """Returns the path of the current log file"""
        return os.path.join(LOGS_DIR, f'{cls.script_name}.log')

    @staticmethod
    def info(message: str = "", to_print=False) -> None:
        """Used for logging general info"""
        logging.info(message)
        if to_print:
            print(message)

    @staticmethod
    def debug(message: str = "", to_print=False) -> None:
        """For debbuging porpousess, may be ignored. shoud be used for high output logs"""
        logging.debug(message)
        if to_print:
            print(message)

    @staticmethod
    def error(message: str = "", to_print=False) -> None:
        """Used for logging errors that have been catched -
         Do not use if the error will crash the code itself."""
        logging.error(message)
        if to_print:
            print(message)

    @staticmethod
    def warning(message: str = "", to_print=False) -> None:
        """Used for logging warnings"""
        logging.warning(message)
        if to_print:
            print(message)

    @staticmethod
    def log_json(json_dict: dict, category: str) -> None:
        """Used to log a json dictionary"""
        if category == "info":
            logging.info(json_dict)
        elif category == "debug":
            logging.debug(json_dict)
        elif category == "error":
            logging.error(json_dict)
        elif category == "warning":
            logging.warning(json_dict)
