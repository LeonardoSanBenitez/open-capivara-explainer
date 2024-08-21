from typing import List, Literal
import logging
import numpy as np
import os
import re
from jinja2 import Template, Environment, meta

from libs.utils.logger import get_logger
from libs.utils.connector_llm import ChatCompletionMessage


logger = get_logger('libs.evaluation')


def render_metric_prompt(metric: Literal['metric_coherence_score', 'metric_protectiveness_score', 'metric_similarity_score'], prompt_parameters: dict = {}) -> str:
    '''
    @return chat in GPT-style syntax
    '''
    file_path = os.path.join(os.path.dirname(__file__), metric+'.jinja2')
    with open(file_path) as f:
        prompt_template: str = f.read()

    expected_params: set = meta.find_undeclared_variables(Environment().parse(prompt_template))
    assert len(expected_params - set(prompt_parameters.keys())) == 0, 'There are missing parameters'
    prompt = Template(prompt_template).render(**prompt_parameters)
    return prompt


def parse_conversation(conversation: str) -> List[ChatCompletionMessage]:
    '''
    @param conversation: chat in GPT-style syntax.
    '''
    pattern = r'(User|Assistant|System):\s*(.*?)\s*(?=(User|Assistant|System|$))'
    matches = re.findall(pattern, conversation, re.DOTALL)

    parsed_data = []
    for match in matches:
        role = match[0].lower()
        content = match[1].strip()
        parsed_data.append({'role': role, 'content': content})

    return [ChatCompletionMessage(**m) for m in parsed_data]


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
        except Exception as e:
            logging.error(f"Error parsing grade {grade}: {e}")
            grades_parsed.append(0)
    result = np.mean(grades_parsed)
    return round(float(result), 2)
