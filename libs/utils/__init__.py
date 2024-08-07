import logging


def setup_loggers():
    # Our logging level
    logging.root.setLevel(logging.INFO)

    # Lift level of third party modules
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'azure' in logger.name:
            logger.setLevel(logging.WARNING)
