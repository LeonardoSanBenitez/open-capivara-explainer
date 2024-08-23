from typing import List, Dict, Union, Tuple
import re
import json
from libs.utils.logger import get_logger


logger = get_logger("utils.json_resilient")


def preprocess_json_input(input_str: str) -> str:
    # Replace single backslashes with double backslashes, while leaving already escaped ones intact
    corrected_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str)
    return corrected_str


def split_string_except_inside_brackets(input_string: str) -> List[str]:
    '''
    Splits the input string by commas, newline characters, and literal '\n' sequences,
    except when they are contained within curly braces {} or square brackets [].

    The function preserves the integrity of JSON-like objects and arrays by not splitting
    them even if they contain delimiters. Escaped backslashes are also handled properly.

    TODO:
    This is not a suuuuper general function
    It can't do its job with the following string:
    assert split_string_except_inside_brackets("abc,def\\nghi,{jkl,mno,pqr},stu\\n\\vwx[yz") == ['abc', 'def', 'ghi', '{jkl,mno,pqr}', 'stu', '\\vwx[yz']
    But it work for most jsonl strings that I'm interested in, so I'll not fix it now

    Parameters:
    input_string (str): The string to be split.

    Returns:
    list: A list of substrings from the input string, split by the specified delimiters unless those delimiters are inside brackets.
    '''
    # Initialize a list to store the split parts
    parts = []
    # Initialize an empty string to build the current part
    current_part = ''
    # Initialize a stack to keep track of brackets
    bracket_stack = []
    # Initialize a variable to keep track of escaped backslashes
    escaped_backslash = False
    # Initialize a pointer to iterate through the string
    i = 0

    while i < len(input_string):
        char = input_string[i]

        # Check if we are at an escaped backslash
        if char == '\\' and not escaped_backslash:
            escaped_backslash = True
            current_part += char
            i += 1
            continue

        # If we encounter a bracket, push it onto the stack
        if (char == '{' or char == '[') and not escaped_backslash:
            bracket_stack.append(char)
        # If we encounter a closing bracket, pop from the stack
        elif (char == '}' and bracket_stack and bracket_stack[-1] == '{') or (char == ']' and bracket_stack and bracket_stack[-1] == '['):
            bracket_stack.pop()

        # If we encounter a delimiter and the stack is empty, split the string
        if (char == ',' or char == '\n' or (char == '\\' and i + 1 < len(input_string) and input_string[i + 1] == 'n')) and not bracket_stack:
            if char == '\\' and input_string[i + 1] == 'n':
                i += 1  # Skip the 'n' after '\'
            parts.append(current_part)
            current_part = ''
        else:
            current_part += char

        # Reset the escaped_backslash flag if it was set
        escaped_backslash = False
        i += 1

    # Add the last part if it's not empty
    if current_part:
        parts.append(current_part)

    return parts


def json_loads_resilient(input_: Union[str, List, Dict]) -> Tuple[bool, Union[List, Dict]]:
    '''
    @return worked: bool
    @return parsed: object
    '''
    if isinstance(input_, list) or isinstance(input_, dict):
        return True, input_
    try:
        parsed = json.loads(input_, strict=False)
    except json.JSONDecodeError:
        preprocessed_text = preprocess_json_input(input_)

        # handle possible jsonl by returning just the first element
        # TODO: maybe should be optional if it should return a list or the first
        splits = split_string_except_inside_brackets(preprocessed_text)
        if len(splits) > 1:
            logger.warning(f"json_loads_resilient: received jsonl with multiple elemnts, returning only the first element...")
        preprocessed_text = splits[0]
        try:
            logger.debug('Trying to parse:', preprocessed_text)
            parsed = json.loads(preprocessed_text, strict=False)
            assert isinstance(parsed, dict) or isinstance(parsed, list)
        except Exception:
            return False, {"Error": f"Could not parse invalid json: {input_}"}
    except TypeError:
        return False, {"Error": f"the JSON object must be str, bytes or bytearray not {type(input_)}"}
    return True, parsed
