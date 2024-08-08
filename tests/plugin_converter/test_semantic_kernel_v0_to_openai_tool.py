import pytest
from libs.plugin_converter.semantic_kernel_v0_to_openai_tool import generate_functions_from_sk_plugins, generate_definitions_from_sk_plugins
from libs.plugins.plugin_capital import PluginCapital


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
def test_generate_functions_empty():
    functions = generate_functions_from_sk_plugins([])
    assert type(functions) == dict
    assert len(functions) == 0

@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
def test_generate_functions_one():
    functions = generate_functions_from_sk_plugins([(PluginCapital(), "PluginCapital")])
    assert type(functions) == dict
    assert len(functions) == 1
    assert all([callable(functions[f]) for f in functions])

@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
def test_generate_definitions_empty():
    definitions = generate_definitions_from_sk_plugins([])
    assert type(definitions) == list
    assert len(definitions) == 1
    assert 'answer' in [d['name'] for d in definitions]

@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Due to pydantic and our introspection
def test_generate_definitions_one():
    definitions = generate_definitions_from_sk_plugins([(PluginCapital(), "PluginCapital")])
    assert type(definitions) == list
    assert len(definitions) == 2
    assert 'answer' in [d['name'] for d in definitions]
    assert any(['get_capital' in d['name'] for d in definitions])
