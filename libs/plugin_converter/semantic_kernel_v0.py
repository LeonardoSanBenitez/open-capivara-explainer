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
