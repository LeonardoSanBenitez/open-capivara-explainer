from libs.utils.logger import get_logger

def test_get_logger():
    logger = get_logger(__name__)
    assert logger is not None