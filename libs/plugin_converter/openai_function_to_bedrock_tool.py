from typing import List
from libs.utils.connector_llm import DefinitionOpenaiFunction, DefinitionBedrockTool, DefinitionBedrockToolSpec, DefinitionBedrockToolInputSchema, ParametersOpenaiFunction


def generate_definitions(functions: List[DefinitionOpenaiFunction]) -> List[DefinitionBedrockTool]:
    outputs = []
    for f in functions:
        outputs.append(DefinitionBedrockTool(
            toolSpec=DefinitionBedrockToolSpec(
                name=f.name,
                description=f.description,
                inputSchema=DefinitionBedrockToolInputSchema(json=f.parameters),
            )
        ))

    assert len(outputs) == len(functions)
    return outputs
