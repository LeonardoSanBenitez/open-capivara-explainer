from typing import List, Tuple, Union, Dict, Optional
from libs.utils.prompt_manipulation import DefinitionOpenaiFunction, DefinitionOpenaiTool


def generate_definitions(functions: List[DefinitionOpenaiFunction]) -> List[DefinitionOpenaiTool]:
    tools = []
    for function in functions:
        tools.append(DefinitionOpenaiTool(type="function", function=function))
    return tools
