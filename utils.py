import logging


def create_logger(name, log_info_file, error_info_file, console_level=logging.INFO, formatter=None):
    if not formatter:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs info messages
    ifh = logging.FileHandler(log_info_file)
    ifh.setLevel(logging.INFO)
    # create file handler which logs from error messages
    efh = logging.FileHandler(error_info_file)
    efh.setLevel(logging.WARNING)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    ch.setFormatter(formatter)
    ifh.setFormatter(formatter)
    efh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(ifh)
    logger.addHandler(efh)

    return logger
