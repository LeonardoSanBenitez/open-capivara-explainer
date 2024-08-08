from typing import Dict, Callable, Optional, List, Any, Literal
from pydantic import BaseModel
import hashlib
from promptflow.tools import aoai   ##TODO!!!!!!!!!
import time

from libs.utils.logger import get_logger
from libs.utils.prompt_manipulation import count_message_tokens, count_string_tokens, create_chat_message, generate_context, construct_prompt
from libs.utils.parse_llm import detect_function_call
from libs.plugin_orchestrator.answer_validation import ValidatedAnswer, Citation, default_answer_validator

logger = get_logger('libs.plugin_orchestrator')

class MaxStepsExceededError(Exception):
    pass


class SecondaryChannelDefinition(BaseModel):
    name: str
    session_id: str
    update_type: Literal['every_step', 'at_finish', 'at_failure']
    send_func: Callable[[Dict[str, Any]], None]  # Any function that follows this signature is accepted


class AutoGPT:
    '''
    # Considerations about output validation
    function that will be applied after the answer is generated.
    Receives an unstructured dict (dependant of the 'answer' definition) and returns a ValidatedAnswer.
    If it raises an exception, this will NOT be caught by the AutoGPT, and will be passed to the caller.
    Serves both as validator and parsing

    # Considerations about multiple/structured outputs
    handled with answer_validator

    # Considerations about token count    
    Parameters token_limit_input and token_limit_output.
    If input and output are counted togehter (lie GPT3), the model needs window size of token_limit_input + token_limit_output.
    If token_limit_output is None, the model can generate as much as is remaining
    '''
    connection: Any  # TODO: type?!
    tools: Dict[str, Callable]
    full_message_history: List[Dict[str, str]]
    definitions: List[dict]
    token_limit_input: int
    token_limit_output: Optional[int]
    max_steps_recommended: int
    max_steps_allowed: int
    called_functions: List[str] = []
    system_prompt: str
    triggering_prompt: str
    user_prompt: str
    model_or_deployment_name: str
    secondary_channels: List[SecondaryChannelDefinition] = []
    answer_validator: Callable[[dict], ValidatedAnswer]

    current_step: int = 1
    definitions_active: List[dict]  # Only the ones the model is allowed to call in the current step
    citations: List[Citation] = []

    def __init__(
        self,
        connection, # anything that exposes the methods <TODO take schema from prompflow>, like for example a promptflow connector
        tools: Dict[str, Callable],
        full_message_history: List[dict],
        definitions: List[dict],
        token_limit_input: int,
        token_limit_output: Optional[int],
        max_steps_recommended: int,
        max_steps_allowed: int,
        system_prompt: str,
        triggering_prompt: str,
        user_prompt: str,
        model_or_deployment_name: str,
        secondary_channels: List[SecondaryChannelDefinition] = [],
        answer_validator: Callable[[dict], ValidatedAnswer] = default_answer_validator
    ):
        self.connection = connection
        self.tools = tools
        self.full_message_history = full_message_history
        self.definitions = definitions
        self.definitions_active = definitions
        self.token_limit_input = token_limit_input
        self.token_limit_output = token_limit_output
        self.max_steps_recommended = max_steps_recommended
        self.max_steps_allowed = max_steps_allowed
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.user_prompt = user_prompt
        self.model_or_deployment_name = model_or_deployment_name
        self.secondary_channels = secondary_channels
        self.answer_validator = answer_validator

    def chat_with_ai(self) -> dict:
        """Interact with the OpenAI API, sending the prompt, message history and functions."""
        next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
            self.system_prompt,
            self.full_message_history,
            self.user_prompt,
        )
        # Account for user input (appended later)
        current_tokens_used += count_message_tokens([create_chat_message("system", self.triggering_prompt)])
        current_tokens_used += 35 + 8*2
        
        # Add Messages until the token limit is reached or there are no more messages to add.
        was_aborted: bool = False
        while next_message_to_add_index >= 0:
            message_to_add = self.full_message_history[next_message_to_add_index]

            tokens_to_add = count_message_tokens([message_to_add])
            logger.debug(f"Token usage: already have {current_tokens_used}, trying to add {tokens_to_add} more")
            if current_tokens_used + tokens_to_add > self.token_limit_input:
                logger.warning(f"Input token limit exceeded before including all history.")
                current_context.insert(insertion_index, create_chat_message("system", "The input token limit was already exceeded. Next you should call 'answer' with all the results you have, even if it is incomplete. Do not call any other function."))
                was_aborted = True
                break
            logger.debug(f'Token usage: {current_tokens_used + tokens_to_add}')

            # Add the most recent message to the start of the current context, after the two system prompts.
            current_context.insert(insertion_index, self.full_message_history[next_message_to_add_index])

            current_tokens_used += tokens_to_add
            next_message_to_add_index -= 1

        # Append user input, the length of this is accounted for above
        current_context.extend([create_chat_message("system", f'This will be your step {self.current_step}. ' + self.triggering_prompt)])
        if was_aborted:
            current_context.extend([create_chat_message("system", "The input token limit was already exceeded. Next you should call 'answer' with all the results you have, even if it is incomplete. Do not call any other function.")])

        current_context_str = construct_prompt(current_context)
        #print('>>>>>> SENDING PROMPT TO AI:', current_context_str)
        # TODO: function calling is deprecated; use tool calling
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice

        # TODO: currently the model is oblied to call a function
        # For agents like ReAct, either:
        #   1. there should be functions like "produce_thought" and "produce_observation"; those steps only add a new message
        #   2. we don't obly the model to call a function
        
        if '-0613' in self.model_or_deployment_name:
            response_format = { "type": "text" }
        else:
            response_format = { "type": "json_object" }

        try:
            response = aoai.chat(
                connection=self.connection,
                prompt=current_context_str,
                deployment_name=self.model_or_deployment_name,
                max_tokens=self.token_limit_output,
                functions=self.definitions_active,
                response_format = response_format,
                temperature = 0.0
            )
            assert type(response) == dict
            return response
        except Exception as e:
            if "The API deployment for this resource does not exist" in str(e):
                raise Exception("Please fill in the deployment name of your Azure OpenAI resource gpt-4 model.")
            else:
                raise e

    def call_function(self, function_name: str, function_arguments: dict) -> str:
        result_function: str = ''  # this is shown to the user through the secondary channels
        result_final: str = ''  # This is what is shown to the LLM, so it has to be nicely formatted
        result_status_code: int = -1
        # Register call
        # TODO: have a class parameter to define if "calling the same function with the same arguments" is allowed
        call_hash = hashlib.md5(f"{function_name}+{function_arguments}".encode('utf-8')).hexdigest()
        if call_hash in self.called_functions:
            logger.warning(f"Function {function_name} was already called with the same arguments.")
            result_final += f"The function {function_name} was already called with the same arguments. Please refer to the previous results, instead of calling the same function twice with the same arguments. If you are not sure how to proceed, call 'answer' with all the results you have (your results are probably good enough, just answer normally).\n"
            result_status_code = 202
        self.called_functions.append(call_hash)

        # Call
        tool = self.tools[function_name]
        try:
            #logger.info(f"Next function = {function_name}, arguments = {function_arguments}")
            result_function += tool(**function_arguments)
            result_final += f"Executed function {function_name} and returned: {result_function}"
            result_status_code = 200 if result_status_code != 202 else 202
        except Exception as e:
            result_function = f"{type(e).__name__}: {str(e)}"
            result_final += f"Failure: function {function_name} raised {type(e).__name__}: {str(e)}"
            result_status_code = 500

        if self.current_step >= self.max_steps_recommended:
            result_final += f"\nAs this was the step {self.current_step - 1}/{self.max_steps_recommended}, next you should call 'answer' with all the results you have."
            self.definitions_active = list(filter(lambda x: x['name'] == 'answer', self.definitions))


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
            self.citations.append(citation)
            # include in the result_final something like: "to reference this result, use the id {citation_id}"


        # Send update to the secondary-channels
        # Should these updates be sent if the deduplication-check fails? If not, add an `if result_status_code != 202`
        for channel in filter(lambda x: x.update_type == 'every_step', self.secondary_channels):
            logger.info(f"Sending update to secondary channel {channel.name}, session {channel.session_id}")
            channel.send_func({
                'step': self.current_step - 1,
                'status_code': result_status_code,
                'function_name': function_name,
                'function_arguments': function_arguments,
                'result': result_function,
                'sessionId': channel.session_id,
                'message': f'At step {self.current_step - 1}, executing {function_name}',  # TODO: the frontend should start using the other fields, and this field should be removed
                'timestamp': int(time.time()),
                'citation_id': citation_id,
            })

        return result_final

    def run(self) -> ValidatedAnswer:
        while True:
            if self.current_step > self.max_steps_allowed:
                logger.error(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")
                raise MaxStepsExceededError(f"Exceeded the maximum number of steps allowed: {self.max_steps_allowed}")

            # Send message to AI, get response
            response = self.chat_with_ai()
            self.current_step += 1

            #print('>>>>>>>>> RAW RESPONSE:', response)
            is_function_calling, function_name, function_arguments = detect_function_call(response)
            if is_function_calling:
                assert type(function_name) == str
                assert type(function_arguments) == dict
                text = f"At step {self.current_step - 1} the function {function_name} was requested, with args = {function_arguments}"
                logger.info(text)
                self.full_message_history.append(create_chat_message("system", text))
                
                if function_name == "answer":
                    # TODO: inform the LLM if the answer is not valid, so it can try again (currently, the validator must handle this, and can't communicate with the LLM)
                    assert 'citations' not in function_arguments, "The 'citations' key is reserved for the answer_validator"
                    function_arguments['citations'] = self.citations
                    validated_answer = self.answer_validator(function_arguments)
                    return validated_answer
                elif function_name in self.tools:
                    command_result = self.call_function(function_name, function_arguments)
                else:
                    command_result = f"Failure: unknown function '{function_name}'. Please refer to available functions defined in functions parameter."

                # Append command result to the message history
                self.full_message_history.append(create_chat_message("system", str(command_result), function_name))
                logger.info(f"result: {command_result}")
            else:
                text = f"At step {self.current_step - 1} the wrong syntax was used to call a function. The wrong syntax used was: {response['content']}"
                logger.warning(text)
                self.full_message_history.append(create_chat_message("system", f"{text}.\n\nPlease try again."))
