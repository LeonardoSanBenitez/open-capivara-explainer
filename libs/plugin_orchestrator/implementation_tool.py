from typing import Dict, Callable, Optional, List, Any, Literal, AsyncGenerator
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import hashlib
import time

from libs.utils.logger import get_logger
from libs.utils.prompt_manipulation import count_message_tokens, count_string_tokens, create_chat_message, generate_context, construct_prompt
from libs.utils.parse_llm import detect_function_call
from libs.plugin_orchestrator.answer_validation import ValidatedAnswer, ValidatedAnswerPart, Citation, default_answer_validator
from libs.utils.connector_llm import ConnectorLLM, ChatCompletionMessage, ChatCompletionMessageResponse
from libs.utils.prompt_manipulation import DefinitionOpenaiTool


'''
TODO:
handle more than one call in a single step

support async tools and async secondary channels
implement considary channel update types "at_finish" and "at_failure"
strong typing for the parameters of send_func, one for each update_type
Maybe also support something for streaming the answer, like update_type=every_chunk_of_answer
'''
logger = get_logger('libs.plugin_orchestrator')


class MaxStepsExceededError(Exception):
    pass


class SecondaryChannelDefinition(BaseModel):
    name: str
    session_id: str
    update_type: Literal['every_step', 'at_finish', 'at_failure']
    send_func: Callable[[Dict[str, Any]], None]  # Any function that follows this signature is accepted


class OrchestratorWithTool(BaseModel):
    '''
    # Considerations about output validation
    function that will be applied after the answer is generated.
    Receives an unstructured dict (dependant of the 'answer' definition) and returns a ValidatedAnswer.
    If it raises an exception, this will NOT be caught by the Orchestrator, and will be passed to the caller.
    Serves both as validator and parsing

    # Considerations about multiple/structured outputs
    handled with answer_validator

    # Considerations about token count
    Parameters token_limit_input and token_limit_output.
    If input and output are counted togehter (lie GPT3), the model needs window size of token_limit_input + token_limit_output.
    If token_limit_output is None, the model can generate as much as is remaining

    # Considerations about Secondary Channels
    Can be used to send data to other services, like a database, a chatbot, or a streaming service.
    Can be used for streaming, as an alternative to `run_stream`.
    Can be used as hooks/callbacks.
    The definition includes a function written by you (the caller), where you can define how to integrate with your service.
    Defined in the parameter `secondary_channels: List[SecondaryChannelDefinition]`.

    '''
    connection: ConnectorLLM
    tool_definitions: List[DefinitionOpenaiTool]
    tool_callables: Dict[str, Callable[..., str]]

    token_limit_input: int
    token_limit_output: Optional[int]
    max_steps_recommended: int
    max_steps_allowed: int

    system_prompt: str
    triggering_prompt: str
    user_prompt: str

    secondary_channels: List[SecondaryChannelDefinition] = []
    answer_validator: Callable[[dict], ValidatedAnswer] = default_answer_validator
    full_message_history: List[ChatCompletionMessage] = []  # Can be set in the initialization to pass history, but as also modified along the way

    _current_step: int = PrivateAttr(default=1)
    _tool_definitions_active: List[DefinitionOpenaiTool] = PrivateAttr(default=[])  # Only the ones the model is allowed to call in the current step
    _citations: List[Citation] = PrivateAttr(default=[])
    _called_functions: List[str] = PrivateAttr(default=[])

    @model_validator(mode='after')
    def _model_validator(self):
        self._tool_definitions_active = self.tool_definitions

    async def _chat_step(self) -> ChatCompletionMessageResponse:
        '''One interaction with the LLM, sending the prompt, message history and functions.'''
        ################################
        # Preprocess prompt
        # This step have no side effect, all operations are just on local variables
        # TODO: This is a very ugly legacy code that requires converting form pydantic to dicts, then back to pydantic
        converted_full_message_history = [m.dict() for m in self.full_message_history]
        next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
            self.system_prompt,
            converted_full_message_history,
            self.user_prompt,
        )
        # Account for user input (appended later)
        current_tokens_used += count_message_tokens([create_chat_message("system", self.triggering_prompt)])
        current_tokens_used += 35 + 8*2

        # Add Messages until the token limit is reached or there are no more messages to add.
        was_aborted: bool = False
        while next_message_to_add_index >= 0:
            message_to_add = converted_full_message_history[next_message_to_add_index]

            tokens_to_add = count_message_tokens([message_to_add])
            logger.debug(f"Token usage: already have {current_tokens_used}, trying to add {tokens_to_add} more")
            if current_tokens_used + tokens_to_add > self.token_limit_input:
                logger.warning(f"Input token limit exceeded before including all history.")
                current_context.insert(insertion_index, create_chat_message(
                    "system",
                    "The input token limit was already exceeded. Next you should call 'answer' with all the results you have, even if it is incomplete. Do not call any other function."
                ))
                was_aborted = True
                break
            logger.debug(f'Token usage: {current_tokens_used + tokens_to_add}')

            # Add the most recent message to the start of the current context, after the two system prompts.
            current_context.insert(insertion_index, converted_full_message_history[next_message_to_add_index])

            current_tokens_used += tokens_to_add
            next_message_to_add_index -= 1

        # Append user input, the length of this is accounted for above
        current_context.extend([create_chat_message("system", f'This will be your step {self._current_step}. ' + self.triggering_prompt)])
        if was_aborted:
            current_context.extend([create_chat_message(
                "system",
                "The input token limit was already exceeded. Next you should call 'answer' with all the results you have, even if it is incomplete. Do not call any other function."
            )])
        current_step_messages: List[ChatCompletionMessage] = [ChatCompletionMessage(**m) for m in current_context]
        # End of preprocess prompt
        ################################

        try:
            chat_completion = await self.connection.chat_completion_async(
                messages=current_step_messages,
                tool_definitions=self._tool_definitions_active,
            )
            assert len(chat_completion.choices) > 0
            assert chat_completion.choices[0].message.tool_calls is not None
            assert len(chat_completion.choices[0].message.tool_calls) > 0
            return chat_completion.choices[0].message

        except Exception as e:
            if "The API deployment for this resource does not exist" in str(e):
                raise Exception("Please fill in the deployment name of your Azure OpenAI resource gpt-4 model.")
            else:
                raise e

    async def _call_function(self, function_name: str, function_arguments: dict) -> str:
        result_function: str = ''  # this is shown to the user through the secondary channels
        result_final: str = ''  # This is what is shown to the LLM, so it has to be nicely formatted
        result_status_code: int = -1

        # Register call
        # TODO: have a class parameter to define if "calling the same function with the same arguments" is allowed
        call_hash = hashlib.md5(f"{function_name}+{function_arguments}".encode('utf-8')).hexdigest()
        if call_hash in self._called_functions:
            logger.warning(f"Function {function_name} was already called with the same arguments.")
            result_final += f"The function {function_name} was already called with the same arguments."\
                            " Please refer to the previous results, instead of calling the same function twice with the same arguments."\
                            " If you are not sure how to proceed, call 'answer' with all the results you have (your results are probably good enough, just answer normally).\n"
            result_status_code = 202
        self._called_functions.append(call_hash)

        # Call
        tool = self.tool_callables[function_name]
        try:
            # logger.info(f"Next function = {function_name}, arguments = {function_arguments}")
            result_function += tool(**function_arguments)
            result_final += f"Executed function {function_name} and returned: {result_function}"
            result_status_code = 200 if result_status_code != 202 else 202
        except Exception as e:
            result_function = f"{type(e).__name__}: {str(e)}"
            result_final += f"Failure: function {function_name} raised {type(e).__name__}: {str(e)}"
            result_status_code = 500

        if self._current_step >= self.max_steps_recommended:
            result_final += f"\nAs this was the step {self._current_step - 1}/{self.max_steps_recommended}, next you should call 'answer' with all the results you have."
            self._tool_definitions_active = list(filter(lambda d: d.function.name == 'answer', self.tool_definitions))

        result_length = count_string_tokens(result_final)
        if result_length + 600 > self.token_limit_input:
            result_final = f"Failure: function {function_name} returned too much output. Do not execute this function again with the same arguments."
            result_status_code = 413

        # Save as citation
        # Should it be saved if the deduplication-check fails? If not, change to `if result_status_code == 200`
        citation_id: Optional[str] = None
        if result_status_code // 100 == 2:
            citation = Citation(
                document = f'Result of function {function_name}',
                content = result_final,
            )
            citation_id = citation.id
            self._citations.append(citation)
            # TODO: include in the result_final something like: "to reference this result, use the id {citation_id}"

        # Send update to the secondary-channels
        # Should these updates be sent if the deduplication-check fails? If not, add an `if result_status_code != 202`
        for channel in filter(lambda x: x.update_type == 'every_step', self.secondary_channels):
            logger.info(f"Sending update to secondary channel {channel.name}, session {channel.session_id}")
            channel.send_func({
                'step': self._current_step - 1,
                'status_code': result_status_code,
                'function_name': function_name,
                'function_arguments': function_arguments,
                'result': result_function,
                'sessionId': channel.session_id,
                'message': f'At step {self._current_step - 1}, executing {function_name}',  # TODO: the frontend should start using the other fields, and this field should be removed
                'timestamp': int(time.time()),
                'citation_id': citation_id,
            })
        assert type(result_final) == str
        return result_final

    async def run(self) -> ValidatedAnswer:
        while True:
            if self._current_step > self.max_steps_allowed:
                logger.error(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")
                raise MaxStepsExceededError(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")

            # Send message to AI, get response
            response: ChatCompletionMessageResponse = await self._chat_step()
            self._current_step += 1

            # print('>>>>>>>>> RAW RESPONSE:', response)
            is_function_calling, function_name, function_arguments = detect_function_call(response)
            if is_function_calling:
                assert type(function_name) == str
                assert type(function_arguments) == dict
                text = f"At step {self._current_step - 1} the function {function_name} was requested, with args = {function_arguments}"
                logger.info(text)
                self.full_message_history.append(ChatCompletionMessage(role='system', content=text))

                if function_name == 'answer':
                    # TODO: inform the LLM if the answer is not valid, so it can try again (currently, the validator must handle this, and can't communicate with the LLM)
                    assert 'citations' not in function_arguments, "The 'citations' key is reserved for the answer_validator"
                    function_arguments['citations'] = self._citations
                    validated_answer = self.answer_validator(function_arguments)
                    return validated_answer
                elif function_name in self.tool_callables:
                    command_result = await self._call_function(function_name, function_arguments)
                else:
                    command_result = f"Failure: unknown function '{function_name}'. Please refer to available functions defined in functions parameter."

                # Append command result to the message history
                assert type(command_result) == str
                self.full_message_history.append(ChatCompletionMessage(role='system', content=command_result))
                logger.info(f"result: {command_result}")
            else:
                text = f"At step {self._current_step - 1} the wrong syntax was used to call a function. The wrong syntax used was: {response.content}"
                logger.warning(text)
                self.full_message_history.append(ChatCompletionMessage(role='system', content=f'{text}.\n\nPlease try again.'))

    async def run_stream(self) -> AsyncGenerator[ValidatedAnswerPart, None]:  # type: ignore
        # TODO
        pass


# Pseudo test
"""
llm = create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
)

max_steps_recommended = 1

orchestrator = OrchestratorWithTool(
    connection = llm,
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
    tool_callables = generate_callables([(PluginCapital(), "PluginCapital")]),
    token_limit_input = 1024,
    token_limit_output = None,
    max_steps_recommended = max_steps_recommended,
    max_steps_allowed = 3,
    system_prompt=f'''you are an AI assistant whose goal is to answer the user question.
    Always use the tools, do not answer without the tools.
    Your decisions must always be made independently without seeking user assistance.
    Every command has a cost, so be smart and efficient. You'll be rewarded if you answer correctly and quickly.
    Aim to complete tasks in the least number of steps. Whenever you are ready to answer, use the function 'answer'.
    Do not take more than {max_steps_recommended} steps to complete a task (including the function 'answer').
    If at step {max_steps_recommended - 1} you don't have all necessary information, answer the best you can with the information at hand.
    Always try to answer with as much useful information as you can.
    ''',
    triggering_prompt='''Determine which next function to use next.
    If you have enough information to answer the question, use the 'answer' function to signal and remember show your results.
    Call only one function per message.
    Do not call the same function with the same arguments more than once, their outputs are deterministic.
    ''',
    user_prompt='Question: What is the capital of brasil?',
)
r = await orchestrator.run()
print(r)
assert type(r.answer) == str
assert len(r.answer) > 4
"""
