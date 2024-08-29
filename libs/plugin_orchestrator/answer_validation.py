from typing import Literal, Optional, List, Any, Tuple
import logging
import uuid
from pydantic import BaseModel, Field

from libs.utils.json_resilient import json_loads_resilient
from libs.utils.html import convert_markdown_to_html
from libs.utils.connector_llm import ChatCompletionMessage


##########################################
# Models
class Citation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description='used to match between citation, intermediate step and text marks')
    document: str
    content: str


class Visualization(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal['python-matplotlib', 'python-streamlit', 'echarts']
    code: Optional[str] = None  # Only for compatibility with old python-based code, maybe should be removed in the future
    config: Optional[dict] = None


class IntermediateResult(BaseModel):
    step: int
    status_code: int
    function_name: str
    function_arguments: dict
    result: str
    message: str
    timestamp: int
    citation_id: Optional[str]


class ValidatedAnswer(BaseModel):
    answer: str
    citations: List[Citation] = []
    visualizations: List[Visualization] = []
    intermediate_results: List[IntermediateResult] = []

    def to_message(self) -> ChatCompletionMessage:
        '''
        Warning: this is an approximate convertion, it may not suite all cases
        '''
        return ChatCompletionMessage(role='assistant', content=self.answer)


class ValidatedAnswerPart(BaseModel):
    answer: Optional[str] = None
    citations: Optional[List[Citation]] = None
    visualizations: Optional[List[Visualization]] = None
    intermediate_results: Optional[List[IntermediateResult]] = None


##########################################
# Non-streaming validators
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


##########################################
# Streaming validators
def _process_chunk(chunk: str, persistent_state: dict) -> Tuple[str, dict]:
    '''
    Process a chunk of data and update the persistent state.
    Extract the content of the key `text`
    '''
    buffer = persistent_state.get('buffer', '') + chunk  # Current buffer content with any previous partial data
    state: Literal['SEARCHING_KEY', 'READING_VALUE', 'FINISHED'] = persistent_state.get('state', 'SEARCHING_KEY')  # Current state of the state machine: SEARCHING_KEY, READING_VALUE, FINISHED
    key_index = persistent_state.get('key_index', 0)  # Index in key_sequence to track matching of the key
    escape = persistent_state.get('escape', False)  # If the last character was an escape character (i.e., backslash)
    unicode_escape = persistent_state.get('unicode_escape', '')  # Accumulated digits for a unicode escape sequence (e.g., \uXXXX)

    key_sequence = '{"text":"'
    output = ''

    i = 0
    while i < len(buffer):
        if state == 'SEARCHING_KEY':
            if buffer[i] == key_sequence[key_index]:
                key_index += 1
                if key_index == len(key_sequence):
                    state = 'READING_VALUE'
                    i += 1  # Move past the last matched character
                    break
                i += 1
            else:
                key_index = 0
                i += 1
            # Trim processed part
            buffer = buffer[i:]
            i = 0
        elif state == 'READING_VALUE':
            if escape:
                if unicode_escape:
                    needed_chars = 4 - len(unicode_escape)
                    remaining = len(buffer) - i
                    to_take = min(needed_chars, remaining)
                    unicode_escape += buffer[i:i + to_take]
                    i += to_take
                    if len(unicode_escape) == 4:
                        try:
                            char = chr(int(unicode_escape, 16))
                            output += char
                        except ValueError:
                            pass  # Handle invalid unicode escape
                        unicode_escape = ''
                        escape = False
                else:
                    esc_char = buffer[i]
                    i += 1
                    if esc_char == 'n':
                        output += '\n'
                    elif esc_char == 'r':
                        output += '\r'
                    elif esc_char == 't':
                        output += '\t'
                    elif esc_char == 'b':
                        output += '\b'
                    elif esc_char == 'f':
                        output += '\f'
                    elif esc_char == 'u':
                        unicode_escape = ''
                    else:
                        output += esc_char
                    if not unicode_escape:
                        escape = False
            else:
                char = buffer[i]
                i += 1
                if char == '\\':
                    escape = True
                elif char == '"':
                    state = 'FINISHED'
                    break
                else:
                    output += char
            # Trim processed part if buffer has been fully consumed
            if i == len(buffer):
                buffer = ''
                i = 0
        elif state == 'FINISHED':
            break

    # Update the persistent state
    persistent_state['buffer'] = buffer[i:]
    persistent_state['state'] = state
    persistent_state['key_index'] = key_index
    persistent_state['escape'] = escape
    persistent_state['unicode_escape'] = unicode_escape

    return output, persistent_state


def default_answer_validator_stream(chunk: str, persistent_state: dict) -> Tuple[ValidatedAnswerPart, dict]:
    # print('>>>>>>>>>>>> RECEIVED CHUNK', chunk)
    chunk_clean, state = _process_chunk(chunk, persistent_state)
    # print('>>>>>>>>>>>> CLEANED CHUNK', chunk_clean)
    return ValidatedAnswerPart(answer=chunk_clean), state
