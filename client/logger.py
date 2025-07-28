
import logging
from constants import LOG_FILE, LOG_ENCODING

class Logger:
    def __init__(self):
        logging.basicConfig(
            filename = LOG_FILE,  # Name of the log file
            encoding = LOG_ENCODING,
            format = '%(message)s',
            level = logging.INFO,  # Minimum logging level to capture (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        )
        self.current_data = ""
    
    def log(self, message):
        logging.info(message)

    def get_filename(self):
        return LOG_FILE
    
    def get_current_data(self):
        return self.current_data
    
    def set_current_data(self, data):
        self.current_data = data