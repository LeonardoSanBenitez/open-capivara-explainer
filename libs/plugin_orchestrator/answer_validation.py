from typing import Literal, Optional
import logging
import uuid
from pydantic import BaseModel, Field

from libs.utils.json_resilient import json_loads_resilient
from libs.utils.html import convert_markdown_to_html
from libs.utils.connector_llm import ChatCompletionMessage

class Citation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description='used to match between citation, intermediate step and text marks')
    document: str
    content: str


class Visualization(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal['python-matplotlib', 'python-streamlit', 'echarts']
    code: Optional[str] = None  # Only for compatibility with old python-based code, maybe should be removed in the future
    config: Optional[dict] = None


class ValidatedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = []
    visualizations: list[Visualization] = []

    def to_message(self) -> ChatCompletionMessage:
        '''
        Warning: this is an approximate convertion, it may not suite all cases
        '''
        return ChatCompletionMessage(role='assistant', content=self.answer)

class ValidatedAnswerPart(BaseModel):
    answer: Optional[str] = None
    citations: Optional[list[Citation]] = None
    visualizations: Optional[list[Visualization]] = None


def default_answer_validator(result: dict) -> ValidatedAnswer:
    assert type(result) == dict, f"Expected the agent to return a dict, got {type(result)}"
    if ('text' in result) and (type(result['text']) == str):
        result['answer'] = convert_markdown_to_html(result['text'])
    else:
        logging.error("The 'text' key is missing from the 'answer' function.")
        result['answer'] = ''

    if ('code' in result) and (type(result['code']) == str):
        result['visualizations'] = [{"type": "python-matplotlib", "code": result['code']}]
    else:
        result['visualizations'] = []

    if 'echarts-definition' in result:
        if (type(result['echarts-definition']) == str) and (result['echarts-definition'] != ''):
            # LLM made a mistake, but acceptable
            worked, config_parsed = json_loads_resilient(result['echarts-definition'])
            if worked:
                assert type(config_parsed) == dict
                result['visualizations'].append({"type": "echarts", "config": config_parsed})
        elif (type(result['echarts-definition']) == dict) and (len(result['echarts-definition']) > 0):
            # expected case
            config = result['echarts-definition']
            result['visualizations'].append({"type": "echarts", "config": config})

    assert 'citations' in result, "The 'citations' key is missing from the 'answer' function."
    return ValidatedAnswer(**result)
