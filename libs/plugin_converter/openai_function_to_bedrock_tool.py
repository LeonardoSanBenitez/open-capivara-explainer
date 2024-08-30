from typing import List
from libs.utils.connector_llm import DefinitionOpenaiFunction, DefinitionBedrockTool, DefinitionBedrockToolSpec, DefinitionBedrockToolInputSchema


def generate_definitions(functions: List[DefinitionOpenaiFunction]) -> List[DefinitionBedrockTool]:
    return [d.to_bedrock_tool() for d in functions]
