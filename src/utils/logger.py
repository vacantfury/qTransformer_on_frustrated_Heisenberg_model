import colorlog
import logging
from .constants import DEFAULT_LOGGER_NAME

def get_logger(name=DEFAULT_LOGGER_NAME, level=logging.INFO):
    logger = colorlog.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        for h in logging.root.handlers[:]:
            logger.root.removeHandler(h)
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(levelname)s]%(asctime)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = True  # Allow logs to propagate to root for file handlers
    return logger