import pytest
from libs.utils.connector_llm import DefinitionBedrockTool
from libs.plugins.plugin_capital import PluginCapital
import libs.plugin_converter.openai_tool_to_bedrock_tool as openai_tool_to_bedrock_tool
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function
import libs.plugin_converter.openai_function_to_tool as openai_function_to_tool

def test_empty():
    r = openai_tool_to_bedrock_tool.generate_definitions([])
    assert type(r) == list
    assert len(r) == 0

@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
def test_capital():
    plugins = [
        (PluginCapital(), "PluginCapital"),
    ]
    openai_function = semantic_kernel_v0_to_openai_function.generate_definitions(plugins)
    openai_tool = openai_function_to_tool.generate_definitions(openai_function)
    r  = openai_tool_to_bedrock_tool.generate_definitions(openai_tool)
    assert type(r) == list
    assert len(r) == 2
    assert all([type(definition) == DefinitionBedrockTool for definition in r])
    assert 'answer' in [d.toolSpec.name for d in r]
    assert any(['get_capital' in d.toolSpec.name for d in r])
