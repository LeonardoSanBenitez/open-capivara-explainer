from typing import Dict, Callable, Optional, List, Any, Literal, AsyncGenerator, Tuple
from pydantic import model_validator, computed_field


from libs.utils.connector_llm import ConnectorLLM, ChatCompletionMessage, ChatCompletionMessageResponse, OpenaiFunctionCall
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.json_resilient import json_loads_resilient
from libs.plugin_orchestrator.implementation_bare import OrchestratorBare, logger



class OrchestratorBare(OrchestratorBare):
    '''
    For the LLMs that do not support function calling, but were finetuned for this specific syntax and toolset.

    We do not explain how to call a function nor which functions are available, but e still have to custom-detect the function calls.

    it is up to the caller to properly manage which-LLM-was-finetuned-for-which-toolset; If the LLM calls a tool that is not available, the orchestrator will raise an error.
    '''
    pass
