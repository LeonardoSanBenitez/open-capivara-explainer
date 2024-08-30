from typing import List
from libs.utils.connector_llm import DefinitionOpenaiTool, DefinitionBedrockTool, DefinitionBedrockToolSpec, DefinitionBedrockToolInputSchema


def generate_definitions(tools: List[DefinitionOpenaiTool]) -> List[DefinitionBedrockTool]:
    return [d.to_bedrock_tool() for d in tools]
