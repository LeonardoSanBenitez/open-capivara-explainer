import pytest
from semantic_kernel import ContextVariables
import semantic_kernel as sk
from libs.plugins.plugin_capital import PluginCapital
from libs.plugin_converter.semantic_kernel_v0_to_openai_tool import make_context


def test_plugin_alone():
    plugin = PluginCapital()
    assert plugin.get_capital(make_context({'country': 'Brazil'})).lower() == 'brasília'

@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
async def test_plugin_with_semantic_kernel():
    kernel = sk.Kernel()
    plugin = PluginCapital()
    plugin_sk = kernel.import_plugin(plugin, "PluginCapital")
    
    result_plugin = await kernel.run(
        plugin_sk["get_capital"],
        input_vars = ContextVariables(variables = {'country': 'Brazil'}),
    )
    result = result_plugin.variables.variables['input']
    assert type(result) == str
    assert result.lower() == 'brasília'
