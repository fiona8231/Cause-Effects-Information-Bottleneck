import logging
import os
from time import strftime


def setup_custom_logger(name):
    """
    This methods creates a custom logging framework.
    :param name: the logger name
    :return: the defined logger
    """

    # create formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # define console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # create logging dir
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # define file handler
    file_handler = logging.FileHandler(strftime("logs/logfile_%M_%H_%d_%m_%Y.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
