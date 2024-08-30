from typing import List
from libs.utils.connector_llm import DefinitionOpenaiTool, DefinitionBedrockTool, DefinitionBedrockToolSpec, DefinitionBedrockToolInputSchema


def generate_definitions(tools: List[DefinitionOpenaiTool]) -> List[DefinitionBedrockTool]:
    outputs = []
    for t in tools:
        assert isinstance(t, DefinitionOpenaiTool)
        assert t.type == 'function'
        outputs.append(DefinitionBedrockTool(
            toolSpec=DefinitionBedrockToolSpec(
                name=t.function.name,
                description=t.function.description,
                inputSchema=DefinitionBedrockToolInputSchema(json=t.function.parameters),
            )
        ))

    assert len(outputs) == len(tools)
    return outputs
