from typing import Dict, Callable, Optional, List, Any, Literal, AsyncGenerator, Tuple, Generator
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import hashlib
import time
import asyncio
import nest_asyncio

from libs.utils.logger import get_logger
from libs.utils.prompt_manipulation import count_message_tokens, count_string_tokens, create_chat_message
from libs.plugin_orchestrator.answer_validation import ValidatedAnswer, ValidatedAnswerPart, Citation, IntermediateResult, default_answer_validator, default_answer_validator_stream
from libs.utils.connector_llm import (
    ConnectorLLM,
    ChatCompletionMessage,
    ChatCompletionMessageResponse,
    OpenaiFunctionCall,
    ChatCompletionMessageResponsePart,
    OpenAIToolCallPart,
    OpenaiFunctionCallPart
)
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.json_resilient import json_loads_resilient


'''

'''
logger = get_logger('libs.plugin_orchestrator')


class MaxStepsExceededError(Exception):
    pass


class SecondaryChannelDefinition(BaseModel):
    name: str
    session_id: str
    update_type: Literal['every_step', 'at_finish', 'at_failure']
    send_func: Callable[[Dict[str, Any]], None]  # Any function that follows this signature is accepted


# TODO: separate this into OrchestratorBase (from which everyone inherits) and OrchestratorWithTool
class OrchestratorWithTool(BaseModel):
    '''
    Used to answer one question.
    Will interact with the LLM in a series of [internal] steps, each step being a message exchange.
    The state for one question (and its multiple steps) is handled by this class.
    For an entire conversation (multiple questions from the user), you need to instantiate this class multiple times.
    The state of an entire conversation is not handled by this class, but by the caller, and is passed via full_message_history.

    # Considerations about output validation
    function that will be applied after the answer is generated.
    Serves both as validator and parsing.
    The validation function do not need to set `intermediate_results`, this will be set by the orchestrator
    If it raises an exception, this will NOT be caught by the Orchestrator, and will be passed to the caller.
    This was suppose to be easy to configure by the caller... but the validator are so complicated and tied to the implemtation.
    of the orchestrator that in practice (I think) the caller will only use the validators provided out of the box.

    ## Non-streaming validator
    Receives an unstructured dict (dependant of the 'answer' definition) and returns a ValidatedAnswer.

    ## Streaming validator
    Recevies the text chunk (as generated by the LLM as an argument for the answer function) and a variable to store any needed state.
    The state is cleaned every start of response.


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

    # Considerations about streaming
    Poorly implemented
    If you can, use the non-streaming response, and add another LLM just to generate the final response (which that one is streaming)
    Or use secondary channels

    # Consideration about the synchronous interface
    This class is optimized for the async methods.
    The sync method have the danger of interfering with your existing event loop (it modifies your loop using nest_asyncio), plus have uncessary overhead; Avoid them if possible.

    # Known limitations
    do not handle abrut stops (finish_reason is not checked)

    422 not handled

    400 not handled.
    It returns something like:
    BadRequestError: Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'filtered': True, 'detected': True}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}}}  # noqa

    when streaming, `citations` is not included in the validated response

    In the streaming default validator, if the key `text` is not provided in the answer function, it returns an empty response (instead of clearly failling, like the non-stream)

    # TODO and possible improvements

    citations and intermediate results are somewhat redundant... no?

    handle more than one call in a single step?

    support async tools and async secondary channels
    implement considary channel update types "at_finish" and "at_failure"
    strong typing for the parameters of send_func, one for each update_type
    Maybe also support something for streaming the answer, like update_type=every_chunk_of_answer

    early_stopping_method: if max_steps_allowed is reached, what to do
    Either 'force' or 'generate'

    currently there is no reasoning step (the LLM returns functionCalls immediatly after the user prompt)... is that ok?

    count_message_tokens and count_string_tokens should be a method of the connection
    '''
    connection: ConnectorLLM

    tool_callables: Dict[str, Callable[..., str]]
    tool_definitions: List[DefinitionOpenaiTool]
    _tool_definitions_active: List[DefinitionOpenaiTool] = PrivateAttr(default=[])  # Only the ones the model is allowed to call in the current step

    token_limit_input: int
    token_limit_output: Optional[int]
    max_steps_recommended: int = Field(..., description="The maximum number of steps recommended to complete the task, including the 'answer' function")
    max_steps_allowed: int
    _current_step: int = PrivateAttr(default=1)

    full_message_history: List[ChatCompletionMessage] = []  # Can be set in the initialization to pass history, but as also modified along the way
    _full_message_history_for_debugging: List[ChatCompletionMessage] = PrivateAttr(default=[])  # Complete and unedited message history, just as was generated by the LLM
    prompt_app_system: str
    prompt_app_user: str
    _prompt_orchestrator_triggering: str = PrivateAttr()
    _prompt_orchestrator_system: str = PrivateAttr()

    secondary_channels: List[SecondaryChannelDefinition] = []
    answer_validator: Callable[[dict], ValidatedAnswer] = default_answer_validator
    answer_validator_stream: Callable[[str, dict], Tuple[ValidatedAnswerPart, dict]] = default_answer_validator_stream
    answer_function_name: str = 'answer'
    _citations: List[Citation] = PrivateAttr(default=[])
    _called_functions: List[str] = PrivateAttr(default=[])  # Just the hash
    _intermediate_results: List[IntermediateResult] = PrivateAttr(default=[])

    @model_validator(mode='after')
    def _model_validator(self):
        self._tool_definitions_active = self.tool_definitions
        self._prompt_orchestrator_system = (
                f"Your goal is to answer the user question, and you can call functions (aka actions) to achieve that.\n"
                f"Your decisions must always be made independently without seeking user assistance.\n"
                f"Every action has a cost, so be smart and efficient. You'll be rewarded if you answer correctly and quickly.\n"
                f"Aim to complete the goal in the least number of steps (aka function calls). Whenever you are ready to answer, use the function '{self.answer_function_name}'.\n"
                f"Do not take more than {self.max_steps_recommended} steps to complete a task (including the function '{self.answer_function_name}').\n"
                f"If at step {self.max_steps_recommended - 1} you don't have all necessary information, answer the best you can with the information at hand.\n"
                f"Always try to answer with as much useful information as you can.\n"
        )

        self._prompt_orchestrator_triggering = (
            f"Determine which next function to use next.\n"
            f"If you have enough information to answer the question, use the '{self.answer_function_name}' function to signal and remember show your results.\n"
            f"Call only one function per message.\n"
            f"Do not call the same function with the same arguments more than once, their outputs are deterministic."
        )

    @staticmethod
    def _allow_nesting_in_current_loop() -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply(loop)
        except RuntimeError as e:
            pass

    @staticmethod
    def _detect_function_call(response: ChatCompletionMessageResponse) -> Tuple[bool, Optional[str], Optional[Dict]]:
        '''
        @return is_function_calling: bool
        @return function_name: Optional[str]
        @return function_arguments: Optional[Dict]

        This step have no side effect
        '''
        assert response.content is None or response.content == '', 'There is something in the response, but only tool calls are expected'
        if response.tool_calls is not None and len(response.tool_calls) > 0:
            logger.info('Function calling detected by: native API feature')
            assert response.tool_calls[0].function is not None
            function_call: OpenaiFunctionCall = response.tool_calls[0].function
            worked, function_arguments = json_loads_resilient(function_call.arguments)
            if not worked:
                logger.warning(f"Could not parse, error: {function_arguments}")
                return False, None, None
            else:
                assert type(function_arguments) == dict
                return True, function_call.name, function_arguments
        else:
            return False, None, None

    def _prepare_prompt_step(self) -> List[ChatCompletionMessage]:
        '''
        Assembles the exact prompt to be sent to the LLM to execute the next step.
        Responsibilities:
        * Puts together system prompts and user prompt
        * Describes the tools that can be used (if necessary; aka if LLM does not support tool calling)
        * Adds the full message history, aware of token limits

        This step have no side effect, all operations are just on the returned variable
        '''
        # TODO: This is a very ugly legacy code that requires converting form pydantic to dicts, then back to pydantic
        converted_full_message_history = [m.dict() for m in self.full_message_history]

        current_context: List[dict] = [
            create_chat_message("system", '# General instructions\n' + self._prompt_orchestrator_system + '\n\n# Asistant-specific instructions\n' + self.prompt_app_system),
            create_chat_message("user", self.prompt_app_user),
        ]
        current_tokens_used: int = count_message_tokens(current_context)
        next_message_to_add_index: int = len(converted_full_message_history) - 1
        insertion_index: int = len(current_context)

        # Account for user input (appended later)
        current_tokens_used += count_message_tokens([create_chat_message("system", self._prompt_orchestrator_triggering)])
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
                    f"The input token limit was already exceeded. Next you should call '{self.answer_function_name}' with all the results you have, even if it is incomplete. Do not call any other function."  # noqa: E501
                ))
                was_aborted = True
                break
            logger.debug(f'Token usage: {current_tokens_used + tokens_to_add}')

            # Add the most recent message to the start of the current context, after the two system prompts.
            current_context.insert(insertion_index, converted_full_message_history[next_message_to_add_index])

            current_tokens_used += tokens_to_add
            next_message_to_add_index -= 1

        # Append user input, the length of this is accounted for above
        current_context.extend([create_chat_message("system", f'This will be your step {self._current_step}. ' + self._prompt_orchestrator_triggering)])
        if was_aborted:
            current_context.extend([create_chat_message(
                "system",
                f"The input token limit was already exceeded. Next you should call '{self.answer_function_name}' with all the results you have, even if it is incomplete. Do not call any other function."  # noqa: E501
            )])
        current_step_messages: List[ChatCompletionMessage] = [ChatCompletionMessage(**m) for m in current_context]
        return current_step_messages

    async def _chat_step(self, current_step_messages: List[ChatCompletionMessage]) -> ChatCompletionMessageResponse:
        '''
        One interaction with the LLM, sending the prompt
        This step calls the LLM, potentially generating costs and other side effects, but do not modify any internal state
        TODO: The only reason this is a separate function is to make `run` reusable for the implementation_bare (where it tool_definitions as to be [])... maybe there is a better way
        @param current_step_messages: The prompt to be sent to the LLM
        '''
        # print('>>>>>>>>> SENDING PROMPT TO AI: ', current_step_messages)
        chat_completion = await self.connection.chat_completion_async(
            messages=current_step_messages,
            tool_definitions=self._tool_definitions_active,
        )
        assert len(chat_completion.choices) > 0
        assert chat_completion.choices[0].message.tool_calls is not None
        assert len(chat_completion.choices[0].message.tool_calls) > 0
        # print('>>>>>>>>> RAW RESPONSE:', chat_completion.choices[0].message)
        return chat_completion.choices[0].message

    async def _chat_step_stream(self, current_step_messages: List[ChatCompletionMessage]) -> AsyncGenerator[ChatCompletionMessageResponsePart, None]:
        '''
        Basically the same as _chat_step
        '''
        chat_completion_stream = self.connection.chat_completion_stream_async(
            messages=current_step_messages,
            tool_definitions=self._tool_definitions_active,
        )
        async for chunk in chat_completion_stream:  # type: ignore  # TODO: mypy says I should use await above... but if I do it doesn't work
            if len(chunk.choices) > 0:
                if chunk.choices[0].message is not None:
                    yield chunk.choices[0].message

    async def _call_function(self, function_name: str, function_arguments: dict) -> str:
        '''
        After this function is called, you can be sure that a new IntermediateResult was appended to self._intermediate_results
        '''
        result_function: str = ''  # this is shown to the user through the secondary channels
        result_final: str = ''  # This is what is shown to the LLM, so it has to be nicely formatted
        result_status_code: int = -1

        # Register call
        # TODO: have a class parameter to define if "calling the same function with the same arguments" is allowed
        call_hash = hashlib.md5(f"{function_name}+{function_arguments}".encode('utf-8')).hexdigest()
        if call_hash in self._called_functions:
            logger.warning(f"Function {function_name} was already called with the same arguments.")
            result_final += f"The function {function_name} was already called with the same arguments."\
                            f" Please refer to the previous results, instead of calling the same function twice with the same arguments."\
                            f" If you are not sure how to proceed, call '{self.answer_function_name}' with all the results you have (your results are probably good enough, just answer normally).\n"
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
            result_final += f"\nAs this was the step {self._current_step - 1}/{self.max_steps_recommended}, next you should call '{self.answer_function_name}' with all the results you have."
            self._tool_definitions_active = list(filter(lambda d: d.function.name == self.answer_function_name, self.tool_definitions))

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

        # Assemble final intermediate result
        intermediate_result = IntermediateResult(
            step=self._current_step - 1,
            status_code=result_status_code,
            function_name=function_name,
            function_arguments=function_arguments,
            result=result_function,
            message=f'At step {self._current_step - 1}, executing {function_name}',  # TODO: the frontend should start using the other fields, and this field should be removed
            timestamp=int(time.time()),
            citation_id=citation_id,
        )
        self._intermediate_results.append(intermediate_result)

        # Send update to the secondary-channels
        # Should these updates be sent if the deduplication-check fails? If not, add an `if result_status_code != 202`
        for channel in filter(lambda x: x.update_type == 'every_step', self.secondary_channels):
            logger.info(f"Sending update to secondary channel {channel.name}, session {channel.session_id}")
            payload = intermediate_result.dict()
            payload['sessionId'] = channel.session_id
            channel.send_func(payload)
        assert type(result_final) == str
        return result_final

    async def run_async(self) -> ValidatedAnswer:
        self._full_message_history_for_debugging += self._prepare_prompt_step()
        while True:
            if self._current_step > self.max_steps_allowed:
                logger.error(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")
                raise MaxStepsExceededError(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")

            # Send message to AI, get response
            current_step_messages = self._prepare_prompt_step()
            response: ChatCompletionMessageResponse = await self._chat_step(current_step_messages)
            self._full_message_history_for_debugging.append(response.to_message())
            self._current_step += 1

            is_function_calling, function_name, function_arguments = self._detect_function_call(response)
            if is_function_calling:
                assert type(function_name) == str
                assert type(function_arguments) == dict
                action_request = f"At step {self._current_step - 1} the function {function_name} was requested, with args = {function_arguments}"
                logger.info(action_request)
                action_request_message = ChatCompletionMessage(role='system', content=action_request)
                self.full_message_history.append(action_request_message)
                self._full_message_history_for_debugging.append(action_request_message)

                if function_name == self.answer_function_name:
                    assert 'citations' not in function_arguments, "The 'citations' key is reserved for the answer_validator"
                    function_arguments['citations'] = self._citations
                    validated_answer = self.answer_validator(function_arguments)
                    validated_answer.intermediate_results = self._intermediate_results
                    # TODO: test if answer is valid; if not, inform the LLM and ask ti to try again (currently, the validator can't communicate with the LLM)
                    self._full_message_history_for_debugging.append(validated_answer.to_message())
                    return validated_answer
                elif function_name in self.tool_callables:
                    action_result = await self._call_function(function_name, function_arguments)
                else:
                    action_result = f"Failure: unknown function '{function_name}'. Please refer to available functions defined in functions parameter."

                # Append action result to the message history
                assert type(action_result) == str
                action_result_message = ChatCompletionMessage(role='system', content=action_result)
                self.full_message_history.append(action_result_message)
                self._full_message_history_for_debugging.append(action_result_message)
                logger.info(f"result: {action_result}")
            else:
                result = f"At step {self._current_step - 1} the wrong syntax was used to call a function. The wrong syntax used was: {response.content}"
                logger.warning(result)
                result_message = ChatCompletionMessage(role='system', content=f'{result}.\n\nPlease try again.')
                self.full_message_history.append(result_message)
                self._full_message_history_for_debugging.append(result_message)

    async def run_stream_async(self) -> AsyncGenerator[ValidatedAnswerPart, None]:
        # TODO: can we share as much code as possible wiht the non stream version?
        self._full_message_history_for_debugging += self._prepare_prompt_step()
        while True:
            if self._current_step > self.max_steps_allowed:
                logger.error(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")
                raise MaxStepsExceededError(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")

            # Send message to AI, get response
            current_step_messages = self._prepare_prompt_step()
            response_stream = self._chat_step_stream(current_step_messages)

            # Accumulation or stream answer
            response_accumulated: ChatCompletionMessageResponsePart = ChatCompletionMessageResponsePart(content='', role='', tool_calls=[])
            validator_persistent_state: Dict[Any, Any] = {}
            async for chunk in response_stream:
                assert type(chunk) == ChatCompletionMessageResponsePart
                assert response_accumulated.content is not None
                assert response_accumulated.role is not None
                assert response_accumulated.tool_calls is not None
                if chunk.content is not None:
                    response_accumulated.content += chunk.content
                if (chunk.role is not None) and (response_accumulated.role not in ['user', 'assistant', 'system']):
                    response_accumulated.role += chunk.role

                if (chunk.tool_calls is not None) and (len(chunk.tool_calls) > 0):
                    # A tool is being called
                    if chunk.tool_calls[0].id is not None:
                        # The tool is new
                        assert chunk.tool_calls[0].type is not None
                        assert chunk.tool_calls[0].function is not None
                        assert chunk.tool_calls[0].function.name is not None
                        assert chunk.tool_calls[0].function.arguments is not None
                        response_accumulated.tool_calls.append(OpenAIToolCallPart(
                            id=chunk.tool_calls[0].id,
                            type=chunk.tool_calls[0].type,  # Assumes `type` if fully set when the function is first called, which may not be true
                            function = OpenaiFunctionCallPart(
                                name=chunk.tool_calls[0].function.name,  # Assumes `name` if fully set when the function is first called, which may not be true
                                arguments=chunk.tool_calls[0].function.arguments,  # This will receive more details later, so we don't assume it is fully set
                            )
                        ))
                        # TODO: if the answer is one single token, I think this answer will never be streamed
                        # Or maybe that happens if the answer is called with an empty string as argument, which I should never happens
                    elif (chunk.tool_calls[0].function is not None) and (chunk.tool_calls[0].function.arguments is not None):
                        # The previous called tool received more arguments details
                        last_called_tool = response_accumulated.tool_calls[-1]
                        assert last_called_tool.function is not None
                        assert last_called_tool.function.name is not None
                        assert last_called_tool.function.arguments is not None
                        last_called_tool.function.arguments += chunk.tool_calls[0].function.arguments
                        if last_called_tool.function.name == self.answer_function_name:
                            validated_answer_chunk, validator_persistent_state = self.answer_validator_stream(chunk.tool_calls[0].function.arguments, validator_persistent_state)
                            if (validated_answer_chunk.answer is not None) and len(validated_answer_chunk.answer) > 0:
                                yield validated_answer_chunk

            # print('>>>>>>>>>>>>>>> FULLY ACCUMULATED ANSWER', response_accumulated)
            response = ChatCompletionMessageResponse(**response_accumulated.dict())
            self._full_message_history_for_debugging.append(response.to_message())
            self._current_step += 1

            is_function_calling, function_name, function_arguments = self._detect_function_call(response)
            if is_function_calling:
                assert type(function_name) == str
                assert type(function_arguments) == dict
                action_request = f"At step {self._current_step - 1} the function {function_name} was requested, with args = {function_arguments}"
                logger.info(action_request)
                action_request_message = ChatCompletionMessage(role='system', content=action_request)
                self.full_message_history.append(action_request_message)
                self._full_message_history_for_debugging.append(action_request_message)

                if function_name == self.answer_function_name:
                    # Response was already streamed, so there is nothing else to be done
                    break
                elif function_name in self.tool_callables:
                    action_result = await self._call_function(function_name, function_arguments)
                    assert len(self._intermediate_results) > 0
                    yield ValidatedAnswerPart(intermediate_results = [self._intermediate_results[-1]])
                else:
                    action_result = f"Failure: unknown function '{function_name}'. Please refer to available functions defined in functions parameter."

                # Append action result to the message history
                assert type(action_result) == str
                action_result_message = ChatCompletionMessage(role='system', content=action_result)
                self.full_message_history.append(action_result_message)
                self._full_message_history_for_debugging.append(action_result_message)
                logger.info(f"result: {action_result}")
            else:
                result = f"At step {self._current_step - 1} the wrong syntax was used to call a function. The wrong syntax used was: {response.content}"
                logger.warning(result)
                result_message = ChatCompletionMessage(role='system', content=f'{result}.\n\nPlease try again.')
                self.full_message_history.append(result_message)
                self._full_message_history_for_debugging.append(result_message)

    def run(self) -> ValidatedAnswer:
        self._allow_nesting_in_current_loop()
        return asyncio.run(self.run_async())

    def run_stream(self) -> Generator[ValidatedAnswerPart, None, None]:
        self._allow_nesting_in_current_loop()

        async def consume_stream():
            # gather results from the async generator
            return [chunk async for chunk in self.run_stream_async()]

        chunks = asyncio.run(consume_stream())
        for chunk in chunks:
            yield chunk


# Pseudo test 1
# 1 tool, openAI with tool
# This requires calls to openAI ($$$)
"""
llm = factory_create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
)

orchestrator = OrchestratorWithTool(
    connection = llm,
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
    tool_callables = generate_callables([(PluginCapital(), "PluginCapital")]),
    token_limit_input = 1024,
    token_limit_output = None,
    max_steps_recommended = 2,
    max_steps_allowed = 3,
    prompt_app_system='You are an AI assistant whose goal is to answer the user question.',
    prompt_app_user='What is the capital of brasil?',
)
r = await orchestrator.run_async()
#r = orchestrator.run()
print(r)
assert type(r.answer) == str
assert len(r.answer) > 4
"""
