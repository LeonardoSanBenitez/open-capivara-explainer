from typing import Dict, Callable, Optional, List, Any, Literal, AsyncGenerator, Tuple
from pydantic import model_validator, computed_field


from libs.utils.connector_llm import ConnectorLLM, ChatCompletionMessage, ChatCompletionMessageResponse, OpenaiFunctionCall
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.json_resilient import json_loads_resilient
from libs.plugin_orchestrator.implementation_tool import OrchestratorWithTool, logger

'''
Do not contain separate steps for reasoninig and actionCall (both are done in the same prompt, sequentially)
'''


class OrchestratorBare(OrchestratorWithTool):
    '''
    For the LLMs that do not support function calling, and you have to explain every detail of the syntax.

    Do not contain separate steps for reasoninig and actionCall (both are done in the same prompt, sequentially).
    '''
    @model_validator(mode='after')
    def _model_validator(self):
        self._tool_definitions_active = self.tool_definitions

        self._prompt_orchestrator_triggering = (
            f"Now call a function with the JSON syntax explained.\n"
            f"Do not do anything else, just call a function.\n"
            f"If you have enough information to answer the question, call the '{self.answer_function_name}' function and pass the answer as a parameter.\n"
            f"This is very important: the argument for the '{self.answer_function_name}' function is the only thing the user will see, all your results should be shown here.\n"
            f"Call only one function per message.\n"
            f"Do not call the same function with the same arguments more than once, their outputs are deterministic."
            f"Call a function with the JSON syntax."
        )

    @computed_field  # type: ignore
    @property
    def _prompt_orchestrator_system(self) -> str:
        # This is computed instead of static because the list of active tools may change during the execution
        return (
                f"Your goal is to answer the user question, and you can call functions (aka commands/actions) to achieve that.\n"
                f"Your decisions must always be made independently without seeking user assistance.\n"
                f"Every action has a cost, so be smart and efficient. You'll be rewarded if you answer correctly and quickly.\n"
                f"Aim to complete the goal in the least number of steps (aka function calls). Whenever you are ready to answer, use the function '{self.answer_function_name}'.\n"
                f"Do not take more than {self.max_steps_recommended} steps to complete a task (including the function '{self.answer_function_name}').\n"
                f"If at step {self.max_steps_recommended - 1} you don't have all necessary information, answer the best you can with the information at hand.\n"
                f"Always try to answer with as much useful information as you can.\n"
                f"\n"
                f"You are allowed to call the following function:\n"
                f"{self._prepare_prompt_function_description(self._tool_definitions_active)}\n"
                f"\n"
                f'''The JSON necessary to call a function contains one key named "thought" (its value have type string, with one short sentence descring why you should take this action), one key named "action_name" (its value have type string, with the name of the function to call), and one key named "args" (its value have type object, with the arguments to pass to the function).\n'''  # noqa: E501
                f'''The JSON keys "thought", "action_name" and "args" should be written in this order.\n'''
                f'''The JSON keys "thought", "action_name" and "args" should always be present.\n'''
                '''Example (just to demonstrate the syntax, this example function does not exist, do not call it): {"thought": "The first step to answer the user question is to use my example function", "action_name": "my_example_function", "args": {"my_param": "something"}}\n'''  # noqa: E501
                f"Remember to call the function only once per message.\n"
                f"Do not answer with anything that is not a json, do not add any extra comment, follow exactly the syntax provided.\n"
                f"Always use the tools, do not answer without the tools.\n"
                f"You are only allowed to use the functions described above, do not call any other function.\n"
                f"You MUST strictly follow those guidelines at every assistant message."
        )

    @staticmethod
    def _detect_function_call(response: ChatCompletionMessageResponse) -> Tuple[bool, Optional[str], Optional[Dict]]:
        print('THIS IS WHAT THE LLM IS TRYING TO DO:')
        print(response)
        if response.content is not None:
            logger.info('Function calling detected by: custom implementation')
            worked, content_json = json_loads_resilient(response.content)
            if not worked:
                logger.warning(f"Could not parse, error: {content_json}")
                return False, None, None
            if type(content_json) != dict:
                logger.warning(f"LLM response is not a dict, but a {type(content_json)}")
                return False, None, None

            # Get function name
            # If name isn't found, return False
            if ('action_name' in content_json) and (type(content_json['action_name']) == str):
                function_name = content_json['action_name']
            elif ('function' in content_json) and (type(content_json['function']) == str):
                function_name = content_json['function']
            elif ('function_call' in content_json) and (type(content_json['function_call']) == str):
                function_name = content_json['function_call']
            elif ('name' in content_json) and (type(content_json['name']) == str):
                function_name = content_json['name']
            else:
                return False, None, None

            if function_name.startswith("functions."):
                function_name = function_name.split("functions.")[1]

            # Get function arguments
            if ('args' in content_json) and (type(content_json['args']) == dict):
                function_arguments = content_json['args']
            elif ('arguments' in content_json) and (type(content_json['arguments']) == dict):
                function_arguments = content_json['arguments']
            elif ('kwargs' in content_json) and (type(content_json['kwargs']) == dict):
                function_arguments = content_json['kwargs']
            else:
                function_arguments = {}

            if function_name.startswith("args."):
                function_name = function_name.split("args.")[1]
            if function_name.startswith("arguments."):
                function_name = function_name.split("arguments.")[1]

            assert type(function_name) == str
            assert len(function_name) > 0
            assert type(function_arguments) == dict
            return True, function_name, function_arguments
        else:
            return False, None, None

    @staticmethod
    def _prepare_prompt_function_description(tool_definitions: List[DefinitionOpenaiTool]) -> str:
        '''
        Assembles the prompt that describes to the LLM which function are available to be called.
        Do not contain leading or trailing whitespaces nor newlines.
        '''
        assert len(tool_definitions) > 0
        function_descriptions: str = ''
        i = 1
        for definition in tool_definitions:
            args: List[str] = []
            properties = definition.function.parameters.properties
            for p in properties:
                str_type = f"Type: {properties[p]['type']}"
                str_required = f"Required: {'yes' if p in definition.function.parameters.required else 'no'}"
                str_default = f". Default: {properties[p]['default']}" if p not in definition.function.parameters.required else ''
                args.append(f"{p} ({properties[p]['description']} {str_type}. {str_required}{str_default})")

            function_description = f"{definition.function.name}: {definition.function.description}.\n  Arguments: {', '.join(args)}"
            function_descriptions += f"{i}. {function_description}\n"
            i += 1
        assert len(tool_definitions) == i - 1
        assert type(function_descriptions) == str
        assert len(function_descriptions) > 0
        return function_descriptions.strip()

    async def _chat_step(self, current_step_messages: List[ChatCompletionMessage]) -> ChatCompletionMessageResponse:
        '''
        One interaction with the LLM, sending the prompt
        This step calls the LLM, potentially generating costs and other side effects, but do not modify any internal state
        @param current_step_messages: The prompt to be sent to the LLM
        '''
        try:
            chat_completion = await self.connection.chat_completion_async(
                messages=current_step_messages,
                tool_definitions=[],  # The tools are set in the _prompt_orchestrator_system, therefore are not needed here
            )
            assert len(chat_completion.choices) > 0
            assert chat_completion.choices[0].message.tool_calls is None
            assert chat_completion.choices[0].message.content is not None
            assert len(chat_completion.choices[0].message.content) > 0
            return chat_completion.choices[0].message

        except Exception as e:
            if "The API deployment for this resource does not exist" in str(e):
                # This happens only with AzureOpenAI LLMs
                # We rewrite because the error is unclear
                # TODO: Maybe this should not be done here...
                raise Exception("Please fill in the deployment name of your Azure OpenAI resource gpt-4 model.")
            else:
                raise e


# GPT3
"""
llm = factory_create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
    hyperparameters={'tool_choice': 'none'}
)

orchestrator = OrchestratorBare(
    connection = llm,
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
    tool_callables = generate_callables([(PluginCapital(), "PluginCapital")]),
    token_limit_input = 1024,
    token_limit_output = None,
    max_steps_recommended = 2,
    max_steps_allowed = 3,
    prompt_app_system='You are an AI assistant whose goal is to answer the user question.',
    prompt_app_user='What is the capital of france?',
)

r = await orchestrator.run()
print(r)
print('\n\n' + '-'*30 + '\n\n')
assert type(r.answer) == str
assert len(r.answer) > 4
print(r.answer)
"""
