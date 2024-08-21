from typing import List, Dict
import logging
import numpy as np


def metric_accuracy(grades: List[str]) -> float:
    '''
    Each grade is "Correct" or "Incorrect"
    '''
    accuracy = round((grades.count("Correct") / len(grades)), 2)
    return accuracy


def metric_mean_stars(stars: List[str]) -> float:
    '''
    Each star is 1, 2, 3, 4 or 5
    '''
    grades_parsed = []
    for grade in stars:
        try:
            grades_parsed.append(int(grade))
        except:
            logging.error(f"Error parsing grade {grade}")
            grades_parsed.append(0)
    result = np.mean(grades_parsed)
    return round(float(result), 2)

##############

from typing import Optional, List
from pydantic import BaseModel
from libs.utils.logger import get_logger


logger = get_logger('libs.evaluation')


