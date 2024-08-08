'''
For now this implementation is only used for the "agent orchestratorion", so many things are specific to it

As we clarify the requirements, the classes AutoGPT and AutoGPTWithAssistant should become interchangeble/the same
'''
import os
import json
from openai import AzureOpenAI
from openai.types.beta.threads import Run, Message
from typing import Dict, Callable, List, Optional, Dict, Tuple
import time


from libs.utils.logger import get_logger
from libs.utils.prompt_manipulation import count_message_tokens, count_string_tokens, create_chat_message, generate_context, construct_prompt
from libs.utils.parse_llm import detect_function_call
from libs.plugin_orchestrator.answer_validation import ValidatedAnswer, Citation, default_answer_validator
from libs.plugin_converter.builtin_plugins import answer_definition_text_and_echart

logger = get_logger('libs.plugin_orchestrator')


class AutoGPTWithAssistant():
    '''
    @param definitions: should follow the Assistants schema

    This was not indented to be used for more than one run...
    It may work, though

    This creates one assistant per run

    # Considerations about multiple/structured outputs
    Exposing a function (like send_echart_visualization) to be called before the end of the run; this saves in-memory in the python object
    '''
    generated_visualizations: List[dict] = []
    max_steps_recommended: int # TODO: these 2 are not used
    max_steps_allowed: int

    current_step = 0

    def __init__(
        self,
        client: AzureOpenAI,
        system_prompt: str,
        model: str,
        max_steps_recommended: int,
        max_steps_allowed: int,
        thread_id: Optional[str] = None,
        tools: Dict[str, Callable] = {},
        definitions: List[dict] = [],
    ):
        assert all(['type' in d for d in definitions]), 'definitions should follow the Assistants schema'
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model
        self.max_steps_recommended = max_steps_recommended
        self.max_steps_allowed = max_steps_allowed
        self.definitions = definitions
        if (thread_id is None) or (thread_id == ''):
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
        else:
            self.thread_id = thread_id


    def answer_validator(self, messages: List[Message], run_id: str) -> ValidatedAnswer:
        answer_text: str = ''
        answer_citations: List[Citation] = []  # TODO: concatenate the citations of every agent call... can we get this just from the messages?

        for message in messages:
            if message.run_id == run_id:
                #for file_id in message.file_ids:
                #    content = self.client.files.content(file_id)
                #    viz = json.loads(content.read().decode())
                #    assert type(viz) == dict
                #    answer_visualizations.append({
                #        "type": "echarts",
                #        "config": viz
                #    })
                for content in message.content:
                    if content.type == "text":
                        # TODO: substitute annotations
                        answer_text = content.text.value + ' ' + answer_text

        validated_answer = ValidatedAnswer(
            answer = answer_text.strip(),
            citations = answer_citations,
            visualizations = self.generated_visualizations,  # type: ignore
        )
                    
        return validated_answer

    def run(self, question: str, ) -> Tuple[str, ValidatedAnswer]:
        '''
        @return threat_id: the ID of the current thread, either created (if none was passed) or the one passed
        @return answer
        '''
        assert self.thread_id is not None

        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=question,
        )

        assistant = self.client.beta.assistants.create(
            name="Orchestrator",
            instructions=self.system_prompt,
            tools=self.definitions,  # type: ignore
            model=self.model,
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=assistant.id,
        )
        try:
            while True:
                self.current_step += 1
                # TODO: check max steps
                # TODO: I think the assistant_wait_step_completion should be here
                logger.info(f"At step {self.current_step}, run status is {run.status}")
                if run.status == "failed":
                    raise RuntimeError("Run failed")  # TODO: better message
                elif run.status == "completed":
                    break
                elif run.status == "requires_action":
                    command_results = []
                    if (run.required_action is not None) and (run.required_action.type == "submit_tool_outputs") and (run.required_action.submit_tool_outputs.tool_calls is not None):
                        tool_calls = run.required_action.submit_tool_outputs.tool_calls
                        for tool_call in tool_calls:
                            if tool_call.type == "function":
                                function_name = tool_call.function.name
                                function_arguments = json.loads(tool_call.function.arguments)
                                logger.info(F"At step {self.current_step}, Running function {function_name} with arguments {function_arguments}")
                                assert type(function_name) == str
                                assert type(function_arguments) == dict

                                command_result: str = ''
                                if function_name == "send_echart_visualization":
                                    logger.info("Generating visualization, args:", function_arguments)
                                    if ('echart_definition' not in function_arguments) and ('series' not in function_arguments) and ('options' not in function_arguments):
                                        # TODO: get rid of this, LLM should call it correctly
                                        command_result += "Error: send_echart_visualization requires an echart_definition argument, try again"
                                    else:
                                        if 'series' in function_arguments:
                                            echart_definition = function_arguments
                                        elif 'options' in function_arguments:
                                            echart_definition = function_arguments['options']
                                        elif ('echart_definition' in function_arguments) and ('options' in function_arguments['echart_definition']):
                                            echart_definition = function_arguments['echart_definition']['options']
                                        elif ('echart_definition' in function_arguments):
                                            # this is the only correct way, how the LLM should call it
                                            echart_definition = function_arguments["echart_definition"]
                                        else:
                                            raise ValueError("this never happen...")
                                        assert type(echart_definition) == dict
                                        self.generated_visualizations.append({"type": "echarts", "config": echart_definition})
                                        command_result += "Visualization generated"
                                elif function_name in self.tools:
                                    command_result += self.tools[function_name](**function_arguments)
                                else:
                                    raise ValueError(f"Function {function_name} not found in tools")
                                command_results.append({"tool_call_id": tool_call.id, "output": command_result})
                                logger.info(F"Function {function_name} returned:\n{command_result}")
                            else:
                                logger.warning(f"Unknown requires_action: {run.required_action}")
                    else:
                        logger.warning(f"Unknown requires_action: {run.required_action}")
                    run = self.client.beta.threads.runs.submit_tool_outputs(thread_id=self.thread_id, run_id=run.id, tool_outputs=command_results)  # type: ignore
                run = assistant_wait_step_completion(self.thread_id, run.id)
                run = self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)  # refresh run status
        finally:
            self.client.beta.assistants.delete(assistant_id = assistant.id)
            
        assert run.status == "completed", f"Run status is {run.status}, did not complete; run_id = {run.id} and thread_id = {self.thread_id} and assistant_id = {assistant.id}"
        validated_answer = self.answer_validator(
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id),  # type: ignore
            run_id = run.id,
        )
        return self.thread_id, validated_answer




###########
# Utils
# TODO: move to the class as static methods? or receive the client as a parameter?
###########
def assistant_inspect_thread(thread_id: str) -> None:
    #response = plugin.ask(make_context({'question': 'hi'}))
    #response
    client = AzureOpenAI(
        api_key=str(os.getenv("AZURE_OPENAI_KEY")),
        api_version="2024-02-15-preview",
        azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT")),
    )
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    ) 

    logger.info("Messages in thread:")
    logger.info(messages.model_dump_json(indent=2))
    # TODO:
    # for every message, substitute message annotations
    # Docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant#message-annotations
    # Example: https://github.com/pdichone/vincibits-study-buddy-knwoledge-retrieval/blob/main/main.py#L86


    logger.info("Steps in the last run:")
    run_id = messages.data[0].run_id
    assert type(run_id) == str
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id, run_id=run_id, order="asc"
    )
    for step in run_steps.data:
        logger.info(step.model_dump_json(indent=2))
        logger.info('-------')


def assistant_wait_step_completion(thread_id: str, run_id: str) -> Run:
    client = AzureOpenAI(
        api_key=str(os.getenv("AZURE_OPENAI_KEY")),
        api_version="2024-02-15-preview",
        azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT")),
    )
    t0 = time.time()
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        # The status can be either queued, in_progress, requires_action, cancelling, cancelled, failed, completed, or expired.
        if run.status in ["completed", "cancelled", "expired", "failed", "requires_action"]:
            logger.info(f"Step completion took {int(time.time() - t0)} seconds")
            return run
        time.sleep(2)
        #print('.', end='')

'''# if an image was saved, display it
import io
from PIL import Image
 data = json.loads(messages.model_dump_json(indent=2))
image_file_id = data['data'][0]['content'][0]['image_file']['file_id']
content = client.files.content(image_file_id)
buffer = io.BytesIO(content.read())
image = Image.open(buffer)
image.show()''';

'''# if a text file was written, read it
data = json.loads(messages.model_dump_json())
file_id = data['data'][0]['file_ids'][0]
content = client.files.content(file_id)
content.read()
''';