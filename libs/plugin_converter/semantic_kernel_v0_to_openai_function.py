from typing import List, Tuple, Callable, Literal, Dict, Optional
import inspect
import logging
import traceback
import os
import uuid
import json
from pydantic import BaseModel, Field

from semantic_kernel.memory.null_memory import NullMemory
from semantic_kernel import KernelContext, ContextVariables
from semantic_kernel.kernel_pydantic import KernelBaseModel

from libs.utils.json_resilient import json_loads_resilient
from libs.utils.logger import get_logger
from libs.utils.html import convert_markdown_to_html
from libs.plugin_converter.builtin_plugins import answer_definition_text
from libs.utils.prompt_manipulation import ParametersOpenaiFunction, DefinitionOpenaiFunction
###############################
# Version check
import importlib.metadata
from packaging import version
required_major_version = '0.5.0'
installed_version = importlib.metadata.version('semantic_kernel')

# Parse the version to handle semantic versioning properly
installed_major_version = version.parse(installed_version).base_version

if installed_major_version != required_major_version:
    raise ImportError(f"semantic_kernel version {required_major_version} is required, but {installed_version} is installed.")
# End of version check
###############################

logger = get_logger(f'libs.converter.{os.path.basename(__file__)}', level = logging.WARNING)


###############################
# Some helpers
def make_context(variables: dict):
    '''
    Used to invoke a kernel function outside a kernel
    '''
    assert type(variables) == dict
    for k, v in variables.items():
        assert type(k) == str, f"Key {k} is not a string"
        assert type(v) in [str, int, float, bool, list, dict], f"Value {v} is not a string, int, float, bool, list or dict"

        # Semantic Kernel context variables must be strings
        # TODO: this is an awful behavior, we just do it to keep compatibility with SemanticKernel
        if (type(v) == dict) or (type(v) == list):
            variables[k] = json.dumps(v)

    variables = {k: str(v) for k, v in variables.items()}
    return KernelContext(
        ContextVariables(variables = variables),
        NullMemory(),
        plugins=None,
    )


def validate_param_string(context: KernelContext, name: str, required: bool = True, simplifications: List[str] = []) -> Optional[str]:
    '''
    Returns a clean string (or None, if the parameter is not required) or raises an error

    Applies simplifications to the value retrieved from the context
    '''
    from libs.utils.parse_llm import map_simplification_to_function  # noqa
    if (not required) and (name not in context.variables):
        return None

    if name not in context.variables:
        raise ValueError(f'Missing parameter {name}')
    if type(context.variables[name]) != str:
        raise ValueError(f'Parameter {name} should be a string')
    value = context.variables[name]
    for simplification in simplifications:
        assert simplification in map_simplification_to_function, f'Unknown simplification {simplification}'
        value = map_simplification_to_function[simplification](value)
    assert type(value) == str
    return value

###############################


def _is_sk_function(func) -> bool:
    decorator_sk = getattr(func, "__kernel_function__", None)
    if (decorator_sk is not None) and (decorator_sk is True):
        return True
    else:
        return False


def _sk_function_get_parameters(func) -> List[dict]:
    # TODO:
    # Some plugins have their input defined with the method `__kernel_function_input_description__`
    # This case is not covered
    # This may cause problems if we import third party plugins
    return getattr(func, "__kernel_function_context_parameters__", [])


def _create_wrapper_function(function_to_be_wrapped: Callable) -> Callable:
    def temp(**kwargs) -> str:
        assert type(kwargs) == dict
        try:
            result = function_to_be_wrapped(make_context(kwargs))
        except ValueError as e:
            logger.warning(f'The function {function_to_be_wrapped.__kernel_function_name__} received invalid parameter: {e}')  # type: ignore
            raise e
        except RuntimeError as e:
            logger.warning(f'The function {function_to_be_wrapped.__kernel_function_name__} raised the exception: {e}')  # type: ignore
            raise e
        except Exception as e:
            logger.warning(f'The function {function_to_be_wrapped.__kernel_function_name__} raised the exception: {e}, {traceback.format_exc()}')  # type: ignore
            raise e
        return result
    return temp


def _generate_param_definition(param: dict) -> dict:
    if param['type'] == 'array':
        return {'type': param['type'], 'items': {'type': 'string'}, 'description': param['description'], 'default': param['default_value']}
    else:
        return {'type': param['type'], 'description': param['description'], 'default': param['default_value']}


def generate_callables(plugins: List[Tuple[KernelBaseModel, str]]) -> Dict[str, Callable]:
    '''
    :param plugins: list of plugins to be used; each element is a tuple of plugin instance and its name
    '''
    functions = {}
    for plugin_instance, plugin_name in plugins:
        for method_name, method in inspect.getmembers(plugin_instance, inspect.ismethod):
            if _is_sk_function(method):
                logger.info(f"Found kernel function {method_name} in plugin {plugin_name}")  # TODO: why is this not printed?
                functions[plugin_name + '_' + method.__kernel_function_name__] = _create_wrapper_function(function_to_be_wrapped = method)  # type: ignore
    return functions


def generate_definitions(
    plugins: List[Tuple[KernelBaseModel, str]],
    answer_definition: Optional[DefinitionOpenaiFunction] = answer_definition_text,
) -> List[DefinitionOpenaiFunction]:
    '''
    :param plugins: list of plugins to be used; each element is a tuple of plugin instance and its name
    :param answer_definition: should be chosen to maximize clarity for the LLM.
    Unecessary complications should be avoided, and handled in the AnswerValidator if needed.
    Can not define a field named `citations`, since this will be added automatically.


    # known limitations
    For parameters of type 'array', openAI requires the item type to be specified, but SemanticKernel have no such concept.
    We define that every array has to be an array of strings.
    References:
    https://community.openai.com/t/schema-for-defining-arrays-in-function/271690/2
    https://github.com/microsoft/semantic-kernel/blob/python-0.5.0.dev/python/semantic_kernel/plugin_definition/kernel_function_context_parameter_decorator.py
    '''
    definitions = []

    for plugin_instance, plugin_name in plugins:
        for method_name, method in inspect.getmembers(plugin_instance, inspect.ismethod):
            if _is_sk_function(method):
                logger.info(f"Found kernel function {method_name} in plugin {plugin_name}")
                function_name = plugin_name + '_' + method.__kernel_function_name__  # type: ignore
                parameters = _sk_function_get_parameters(method)
                definitions.append(DefinitionOpenaiFunction(**{
                    "name": function_name,
                    "description": method.__kernel_function_description__,  # type: ignore
                    "parameters": {
                        "type": "object",
                        "properties": {param['name']: _generate_param_definition(param) for param in parameters},
                        "required": [param['name'] for param in parameters if param['required']],
                    },
                }))

    if answer_definition is not None:
        definitions.append(answer_definition)

    return definitions
