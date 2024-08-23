from typing import Literal, List, Tuple, Optional, Any, Union, AsyncGenerator, Generator
from pydantic import BaseModel, ConfigDict, computed_field
from functools import cached_property
from abc import ABC, abstractmethod
import json
import openai
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.logger import get_logger

logger = get_logger('libs.connector_llm')

'''
TODO:

This is an openai-like client for calling LLMs: it behaves exactly/similar to the openai python client, but it supports other providers.

Currently supported providers:
* OpenAI
* Azure openAI
* LlamaCPP
* Bedrock (work in progress)


# Known limitations

text completion method are still not implemented

422 not handled

filtered response not handled

Incomplete mapping model->capabilities

Incomplete hyperparameter support

Tool calling only supported at OpenAI (even though bedrock supports it too)

network failures or other random failures
'''


class Capabilities(BaseModel):
    response_json_only: bool = False
    tool_call: bool = False
    local: bool = False

###


class Hyperparameters(BaseModel, ABC):
    pass


class HyperparametersOpenAI(Hyperparameters):
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    tool_choice: Union[str, dict] = 'required'  # 'required', 'none', or a dict like {"type": "function", "function": {"name": "my_function"}}


class HyperparametersLlamaCPP(Hyperparameters):
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class HyperparametersBedrock(Hyperparameters):
    maxTokens: int = 1024
###


class Credentials(BaseModel, ABC):
    pass


class CredentialsOpenAI(Credentials):
    base_url: str
    api_key: str


class CredentialsBedrock(Credentials):
    '''
    If None, authentication will be done with the current identity
    '''
    region_name: str = 'eu-central-1'
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


###


class ChatCompletionMessage(BaseModel):
    role: str = 'assistant'
    content: str

    def to_bedrock_normal(self) -> dict:
        return {
            'role': self.role,
            'content': [{'text': self.content}]
        }

    def to_bedrock_system(self) -> dict:
        return {
            'text': self.content
        }


#############################################
# Models - Text completion response
class TextCompletion(BaseModel):
    content: str
    stop: bool = True


class TextCompletionPart(BaseModel):
    content: Optional[str] = None
    stop: Optional[bool] = None


#############################################
# Models - Full chat completion response
class OpenaiFunctionCall(BaseModel):
    arguments: str
    name: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal['function']
    function: OpenaiFunctionCall


class ChatCompletionMessageResponse(BaseModel):
    content: Optional[str] = None
    role: str = 'assistant'
    tool_calls: Optional[List[OpenAIToolCall]] = None

    def to_message(self) -> ChatCompletionMessage:
        '''
        Warning: this is an approximate convertion, it may not suite all cases
        '''
        message: ChatCompletionMessage
        if (self.content is None) and (self.tool_calls is None):
            logger.warning('Neither content nor tool_calls are provided. Response will be empty.')
            message = ChatCompletionMessage(role=self.role, content='')
        elif (self.content is not None) and (self.tool_calls is None):
            message = ChatCompletionMessage(role=self.role, content=self.content)
        elif (self.content is None) and (self.tool_calls is not None):
            message = ChatCompletionMessage(role=self.role, content=json.dumps([t.dict() for t in self.tool_calls]))
        else:
            assert type(self.content) == str
            if len(self.content) > 0:
                logger.warning('Both content and tool_calls are not None. Response will contain only content.')
            message = ChatCompletionMessage(role=self.role, content=self.content)

        return message


class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessageResponse
    finish_reason: str


class ChatCompletion(BaseModel):
    choices: List[ChatCompletionChoice]


#############################################
# Models - Partial chat completion response
# Used for streaming
# The only difference is that everything is optional
class OpenaiFunctionCallPart(BaseModel):
    arguments: Optional[str] = None
    name: Optional[str] = None


class OpenAIToolCallPart(BaseModel):
    id: Optional[str] = None
    type: Optional[Literal['function']] = 'function'
    function: Optional[OpenaiFunctionCallPart]


class ChatCompletionMessageResponsePart(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = 'assistant'
    tool_calls: Optional[List[OpenAIToolCallPart]] = None


class ChatCompletionChoicePart(BaseModel):
    message: Optional[ChatCompletionMessageResponsePart]
    finish_reason: Optional[str] = None


class ChatCompletionPart(BaseModel):
    choices: List[ChatCompletionChoicePart]


#############################################
class ConnectorLLM(BaseModel, ABC):
    capabilities: Capabilities
    hyperparameters: Hyperparameters
    credentials: Credentials

    @abstractmethod
    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        pass

    @abstractmethod
    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[ChatCompletionPart, None, None]:
        pass

    @abstractmethod
    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        pass

    @abstractmethod
    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[ChatCompletionPart, None]:
        pass

    @abstractmethod
    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        pass

    @abstractmethod
    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[TextCompletionPart, None, None]:
        pass

    @abstractmethod
    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        pass

    @abstractmethod
    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[TextCompletionPart, None]:
        pass


class ConnectorLLMBedrock(BaseModel, ABC):
    capabilities: Capabilities
    hyperparameters: HyperparametersBedrock
    credentials: Credentials
    modelname: str

    @computed_field  # type: ignore
    @cached_property
    def client(self) -> Any:  # -> botocore.client.BedrockRuntime:
        import boto3  # type: ignore  # noqa
        return boto3.client(
            'bedrock-runtime',
            **self.credentials.dict()
        )

    def _separate_messages(self, messages: List[ChatCompletionMessage]) -> Tuple[List[dict], List[dict]]:
        normal_messages = []
        system_messages = []
        for message in messages:
            if message.role == 'system':
                system_messages.append(message.to_bedrock_system())
            elif message.role == 'user' or message.role == 'assistant':
                normal_messages.append(message.to_bedrock_normal())
            else:
                raise ValueError(f'Unkown message role: {message.role}')
        return normal_messages, system_messages

    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        normal_messages, system_messages = self._separate_messages(messages)
        # print('>>>>>>>>>>>>>', normal_messages)
        # print('>>>>>>>>>>>>>', system_messages)
        chat_completion = self.client.converse(
            modelId=self.modelname,
            messages=normal_messages,
            system=system_messages,
            inferenceConfig=self.hyperparameters.dict(),
        )
        return ChatCompletion(
            choices = [ChatCompletionChoice(
                message = ChatCompletionMessageResponse(
                    role = 'assistant',
                    content = chat_completion['output']['message']['content'][0]['text'],
                ),
                finish_reason = chat_completion['stopReason']
            )]
        )

    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[ChatCompletionPart, None, None]:
        normal_messages, system_messages = self._separate_messages(messages)
        chat_completion_stream = self.client.converse_stream(
            modelId=self.modelname,
            messages=normal_messages,
            system=system_messages,
            inferenceConfig=self.hyperparameters.dict(),
        )
        s = chat_completion_stream.get('stream')
        assert s is not None
        for event in s:
            yield ChatCompletionPart(
                choices = [ChatCompletionChoicePart(
                    message = ChatCompletionMessageResponsePart(
                        role = 'assistant',
                        content = event.get('contentBlockDelta', {}).get('delta', {}).get('text', None),
                    ),
                    finish_reason = event.get('messageStop', {}).get('stopReason', None)
                )]
            )


class ConnectorLLMOpenAI(ConnectorLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    modelname: str
    hyperparameters: HyperparametersOpenAI
    credentials: CredentialsOpenAI

    @computed_field  # type: ignore
    @cached_property
    def response_format(self) -> dict:
        # TODO: currently unused; hasto be set in hyperparam?
        if self.capabilities.response_json_only:
            return {"type": "json_object"}
        else:
            return {"type": "text"}

    @computed_field  # type: ignore
    @cached_property
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.credentials.base_url,
            api_key=self.credentials.api_key,
        )

    @computed_field  # type: ignore
    @cached_property
    def client_async(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=self.credentials.base_url,
            api_key=self.credentials.api_key,
        )

    def _chat_completion_assemble_params(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> dict:
        params = {
            "model": self.modelname,
            "messages": [m.dict() for m in messages],
            **self.hyperparameters.dict(),
        }
        if len(tool_definitions) > 0:
            if self.capabilities.tool_call:
                params['tools'] = [d.dict() for d in tool_definitions]
            else:
                logger.warning("Tool calls are not supported for this model")
        else:
            if 'tool_choice' in params:
                if params['tool_choice'] != 'none':
                    logger.warning('Tool required, but no tool was provided. Overwriting to none.')
                params.pop('tool_choice')
        return params

    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        chat_completion = self.client.chat.completions.create(stream=False, **self._chat_completion_assemble_params(messages, tool_definitions))
        return ChatCompletion(**chat_completion.dict())

    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[ChatCompletionPart, None, None]:
        chat_completion_stream = self.client.chat.completions.create(stream=True, **self._chat_completion_assemble_params(messages, tool_definitions))

        for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages_out: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages_out = [{**m, 'role': m.get('role') or 'assistant'} for m in messages_out]
                messages_out = [{**m, 'content': m.get('content') or ''} for m in messages_out]
                choices: List[ChatCompletionChoicePart] = [ChatCompletionChoicePart(message = ChatCompletionMessageResponsePart(**m)) for m in messages_out]
                yield ChatCompletionPart(choices=choices)

    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        chat_completion = await self.client_async.chat.completions.create(stream=False, **self._chat_completion_assemble_params(messages, tool_definitions))
        return ChatCompletion(**chat_completion.dict())

    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[ChatCompletionPart, None]:  # type: ignore
        chat_completion_stream = await self.client_async.chat.completions.create(stream=True, **self._chat_completion_assemble_params(messages, tool_definitions))
        async for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages_d: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages_d = [{**m, 'role': m.get('role') or 'assistant'} for m in messages_d]
                messages_d = [{**m, 'content': m.get('content') or ''} for m in messages_d]
                choices: List[ChatCompletionChoicePart] = [ChatCompletionChoicePart(message = ChatCompletionMessageResponsePart(**m)) for m in messages_d]
                yield ChatCompletionPart(choices=choices)

    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        raise NotImplementedError()

    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[TextCompletionPart, None, None]:
        raise NotImplementedError()

    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        raise NotImplementedError()

    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[TextCompletionPart, None]:
        raise NotImplementedError()


class ConnectorLLMAzureOpenAI(ConnectorLLMOpenAI):
    @computed_field  # type: ignore
    @cached_property
    def client(self) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(
            azure_endpoint=self.credentials.base_url,
            api_key=self.credentials.api_key,
            api_version="2024-05-01-preview",
        )

    @computed_field  # type: ignore
    @cached_property
    def client_async(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI(
            azure_endpoint=self.credentials.base_url,
            api_key=self.credentials.api_key,
            api_version="2024-05-01-preview",
        )


class ConnectorLLMLlamaCPP(ConnectorLLMOpenAI):
    hyperparameters: HyperparametersLlamaCPP  # type: ignore


def factory_create_connector_llm(
    credentials: Credentials,
    hyperparameters: dict = {},
    provider: Literal['openai', 'azure_openai', 'llama_cpp', 'bedrock'] = 'openai',
    modelname: str = 'gpt-3.5-turbo',
    version: str = '1106',
) -> ConnectorLLM:
    '''
    Chooses the right connector

    set capabilities based on version and model
    '''
    if provider == 'openai':
        capabilities = Capabilities()
        if version == '1106' or version == '0125':
            capabilities.response_json_only = True
            capabilities.tool_call = True

        return ConnectorLLMOpenAI(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersOpenAI(**hyperparameters),
            credentials=credentials,  # type: ignore
        )
    elif provider == 'azure_openai':
        capabilities = Capabilities()
        if version == '1106' or version == '0125':
            capabilities.response_json_only = True
            capabilities.tool_call = True

        return ConnectorLLMAzureOpenAI(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersOpenAI(**hyperparameters),
            credentials=credentials,  # type: ignore
        )
    elif provider == 'llama_cpp':
        capabilities = Capabilities(local=True)
        return ConnectorLLMLlamaCPP(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersLlamaCPP(**hyperparameters),
            credentials=credentials,  # type: ignore
        )
    elif provider == 'bedrock':
        # Actually, tool_call IS supported
        # We just had to adapt the syntax
        capabilities = Capabilities()
        return ConnectorLLMBedrock(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersBedrock(**hyperparameters),
            credentials=credentials,  # type: ignore
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Pseudo integration tests 1A - Simple messages - OpenAI
# this requires calls to openAI ($$$)
'''
llm = factory_create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
    hyperparameters = {'max_tokens': 3, 'tool_choice': 'none'}
)


# chat completion, normal
chat_completion = llm.chat_completion(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
assert type(chat_completion.choices[0].message.content) == str
assert len(chat_completion.choices[0].message.content) > 2
print(chat_completion.choices[0].message.content)

# chat completion, stream
chat_completion_stream = llm.chat_completion_stream(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
answer = ''
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        assert type(chunk.choices[0].message.content) == str
        print(chunk.choices[0].message.content, end='')
        answer += chunk.choices[0].message.content
assert len(answer) > 2

# chat completion, normal, async
chat_completion = await llm.chat_completion_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
assert type(chat_completion.choices[0].message.content) == str
assert len(chat_completion.choices[0].message.content) > 2
print(chat_completion.choices[0].message.content)

# chat completion, stream, async
chat_completion_stream = llm.chat_completion_stream_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
answer = ''
async for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        assert type(chunk.choices[0].message.content) == str
        print(chunk.choices[0].message.content, end='')
        answer += chunk.choices[0].message.content
assert len(answer) > 2
'''


# Pseudo integration tests 1B - Simple messages - LlamaCPP
'''
llm = factory_create_connector_llm(
    provider='llama_cpp',
    modelname='not-needed',
    version='not-needed',
    credentials=CredentialsOpenAI(
        base_url='http://llm-server:8080/v1',
        api_key='not-needed',
    ),
    hyperparameters = {'max_tokens': 10}
)

# chat completion, normal
chat_completion = llm.chat_completion(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
assert type(chat_completion.choices[0].message.content) == str
assert len(chat_completion.choices[0].message.content) > 2
print(chat_completion.choices[0].message.content)

# chat completion, stream
chat_completion_stream = llm.chat_completion_stream(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
answer = ''
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        assert type(chunk.choices[0].message.content) == str
        print(chunk.choices[0].message.content, end='')
        answer += chunk.choices[0].message.content
assert len(answer) > 2

# chat completion, normal, async
chat_completion = await llm.chat_completion_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
assert type(chat_completion.choices[0].message.content) == str
assert len(chat_completion.choices[0].message.content) > 2
print(chat_completion.choices[0].message.content)

# chat completion, stream, async
chat_completion_stream = llm.chat_completion_stream_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
answer = ''
async for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        assert type(chunk.choices[0].message.content) == str
        print(chunk.choices[0].message.content, end='')
        answer += chunk.choices[0].message.content
assert len(answer) > 2
'''


# Pseudo integration tests 1C - Simple messages - Bedrock
# this requires calls to AWS ($$$)
'''
llm = factory_create_connector_llm(
    provider='bedrock',
    modelname='anthropic.claude-3-5-sonnet-20240620-v1:0',
    credentials=CredentialsBedrock(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_ACCESS_KEY_SECRET"),
    ),
)

# chat completion, normal
chat_completion = llm.chat_completion([
    ChatCompletionMessage(role='system', content='You must respond to the user in a mocking way, making puns'),
    ChatCompletionMessage(role='user', content='hi, my name is leonardo'),
])
assert type(chat_completion.choices[0].message.content) == str
assert len(chat_completion.choices[0].message.content) > 2
print(chat_completion.choices[0].message.content)

# chat completion, stream
chat_completion_stream = llm.chat_completion_stream([
    ChatCompletionMessage(role='system', content='You must respond to the user in a mocking way, making puns'),
    ChatCompletionMessage(role='user', content='hi, my name is leonardo'),
])

answer = ''
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        message = chunk.choices[0].message
        if message.content:
            print(message.content, end='')
            answer += message.content
            assert type(message.content) == str
assert len(answer) > 2
'''

# Pseudo integration tests 2A - Tool calling - OpenAI
# this requires calls to openAI ($$$)
'''
import libs.plugin_converter.openai_function_to_tool as openai_function_to_tool
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function
from libs.plugins.plugin_capital import PluginCapital


llm = factory_create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
)

# chat completion, normal, with function call
chat_completion = llm.chat_completion(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
print(chat_completion.choices[0].message.tool_calls[0].dict())
assert len(chat_completion.choices[0].message.tool_calls) > 0
assert 'PluginCapital' in chat_completion.choices[0].message.tool_calls[0].function.name

# chat completion, stream, with function call
chat_completion_stream = llm.chat_completion_stream(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
name=None
arguments=''
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message, end='\n-----\n')
        if chunk.choices[0].message.tool_calls and len(chunk.choices[0].message.tool_calls) > 0:
            if (name is None) and (chunk.choices[0].message.tool_calls[0].function.name is not None):
                name = chunk.choices[0].message.tool_calls[0].function.name
            if chunk.choices[0].message.tool_calls[0].function.arguments is not None:
                arguments += chunk.choices[0].message.tool_calls[0].function.arguments
assert 'PluginCapital' in name
assert 'brazil' in arguments.lower()

# chat completion, normal, async
chat_completion = await llm.chat_completion_async(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
print(chat_completion.choices[0].message.tool_calls[0].dict())
assert len(chat_completion.choices[0].message.tool_calls) > 0
assert 'PluginCapital' in chat_completion.choices[0].message.tool_calls[0].function.name

# chat completion, stream, async
chat_completion_stream = llm.chat_completion_stream_async(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
name=None
arguments=''
async for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message, end='\n-----\n')
        if chunk.choices[0].message.tool_calls and len(chunk.choices[0].message.tool_calls) > 0:
            if (name is None) and (chunk.choices[0].message.tool_calls[0].function.name is not None):
                name = chunk.choices[0].message.tool_calls[0].function.name
            if chunk.choices[0].message.tool_calls[0].function.arguments is not None:
                arguments += chunk.choices[0].message.tool_calls[0].function.arguments
assert 'PluginCapital' in name
assert 'brazil' in arguments.lower()
'''
