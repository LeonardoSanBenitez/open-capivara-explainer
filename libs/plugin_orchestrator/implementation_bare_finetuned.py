from typing import Dict, Callable, Optional, List, Any, Literal, AsyncGenerator, Tuple
from pydantic import model_validator, computed_field

from libs.utils.connector_llm import ConnectorLLM, ChatCompletionMessage, ChatCompletionMessageResponse, OpenaiFunctionCall
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.json_resilient import json_loads_resilient
from libs.plugin_orchestrator.implementation_bare import OrchestratorBare, logger


class OrchestratorBareFinetuned(OrchestratorBare):
    '''
    For the LLMs that do not support function calling, but were finetuned for this specific syntax and toolset.

    We do not explain how to call a function nor which functions are available, but e still have to custom-detect the function calls.

    it is up to the caller to properly manage which-LLM-was-finetuned-for-which-toolset; If the LLM calls a tool that is not available, the orchestrator will raise an error.
    '''
    @model_validator(mode='after')
    def _model_validator(self):
        self._tool_definitions_active = self.tool_definitions

        self._prompt_orchestrator_triggering = (
            # f"Now call a function with the JSON syntax explained.\n"
            # f"Do not do anything else, just call a function.\n"
            # f"If you have enough information to answer the question, call the '{self.answer_function_name}' function and pass the answer as the argument 'text'.\n"
            # f"This is very important: the argument for the '{self.answer_function_name}' function is the only thing the user will see, all your results should be shown here.\n"
            # f"Call only one function per message.\n"
            # f"Do not call the same function with the same arguments more than once, their outputs are deterministic."
            f"Call ONE function with the JSON syntax, nothing else."
        )

    @computed_field  # type: ignore
    @property
    def _prompt_orchestrator_system(self) -> str:
        # This is computed instead of static because the list of active tools may change during the execution
        return (
                f"Your goal is to answer the user question, and you can call functions (aka commands/actions) to achieve that.\n"
                # f"Your decisions must always be made independently without seeking user assistance.\n"
                # f"Every action has a cost, so be smart and efficient. You'll be rewarded if you answer correctly and quickly.\n"
                # f"Aim to complete the goal in the least number of steps (aka function calls). Whenever you are ready to answer, use the function '{self.answer_function_name}'.\n"
                f"Do not take more than {self.max_steps_recommended} steps to complete a task (including the function '{self.answer_function_name}').\n"
                # f"If at step {self.max_steps_recommended - 1} you don't have all necessary information, answer the best you can with the information at hand.\n"
                # f"Always try to answer with as much useful information as you can.\n"
                # f"\n"
                f"You are allowed to call the following function:\n"
                f"{self._prepare_prompt_function_description(self._tool_definitions_active)}\n"
                # f"\n"
                f'''The JSON necessary to call a function contains one key named "thought" (its value have type string, with one short sentence descring why you should take this action), one key named "action_name" (its value have type string, with the name of the function to call), and one key named "args" (its value have type object, with the arguments to pass to the function).\n'''  # noqa: E501
                f'''The JSON keys "thought", "action_name" and "args" should be written in this order.\n'''
                f'''The JSON keys "thought", "action_name" and "args" should always be present.\n'''
                # # if self.prompt_include_example = True
                # # '''Example (just to demonstrate the syntax, this example function does not exist, do not call it): {"thought": "The first step to answer the user question is to use my example function", "action_name": "my_example_function", "args": {"my_param": "something"}}\n'''  # noqa: E501
                f"Remember to call the function only once per message.\n"
                f"Do not answer with anything that is not a json, do not add any extra comment, follow exactly the syntax provided.\n"
                # f"Always use the tools, do not answer without the tools.\n"
                # f"You are only allowed to use the functions described above, do not call any other function.\n"
                f"You MUST strictly follow those guidelines at every assistant message."
        )
