from typing import Literal, List, Optional, AsyncGenerator, Generator, Awaitable
from pydantic import BaseModel, ConfigDict, computed_field
from functools import cached_property
from abc import ABC, abstractmethod
import openai
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.utils.logger import get_logger

logger = get_logger('libs.connector_llm')

'''
TODO:

this is quite specific to the Orchestrator, maybe it should be moved there

text completion method are still not supported

422 not handled

filtered response not handled



converter: convert functionCall to ToolCall

probably the ToolCall is more reliable, so this module should be refactored to deal with it
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
    tool_choice: Optional[str] = "required"


class HyperparametersLlamaCPP(Hyperparameters):
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

###


class Credentials(BaseModel, ABC):
    pass


class CredentialsOpenAI(Credentials):
    base_url: str
    api_key: str

###


class TextCompletion(BaseModel):
    content: str
    stop: bool = True


class ChatCompletionMessage(BaseModel):
    role: str = 'assistant'
    content: str


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


class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessageResponse
    finish_reason: str


class ChatCompletion(BaseModel):
    choices: List[ChatCompletionChoice]


#############################################
# Models - Partial chat completion response
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
    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[ChatCompletion, None, None]:
        pass

    @abstractmethod
    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        pass

    @abstractmethod
    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[ChatCompletion, None]:
        pass

    @abstractmethod
    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        pass

    @abstractmethod
    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[TextCompletion, None, None]:
        pass

    @abstractmethod
    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Awaitable[TextCompletion]:
        pass

    @abstractmethod
    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[TextCompletion, None]:
        pass


class ConnectorLLMOpenAI(ConnectorLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    modelname: str
    hyperparameters: HyperparametersOpenAI
    credentials: CredentialsOpenAI

    @computed_field  # type: ignore
    @cached_property
    def response_format(self) -> dict:
        if self.capabilities.response_json_only:
            return {"type": "json_object"}  # TODO: currently unused; hasto be set in hyperparam?
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
        return params

    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        chat_completion = self.client.chat.completions.create(stream=False, **self._chat_completion_assemble_params(messages, tool_definitions))
        return ChatCompletion(**chat_completion.dict())

    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[ChatCompletionPart, None, None]:  # type: ignore
        chat_completion_stream = self.client.chat.completions.create(stream=True, **self._chat_completion_assemble_params(messages, tool_definitions))

        for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages_out: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages_out = [{**m, 'role': m.get('role') or 'assistant'} for m in messages_out]
                messages_out = [{**m, 'content': m.get('content') or ''} for m in messages_out]
                choices: List[dict] = [{'message': m} for m in messages_out]
                yield ChatCompletionPart(choices=choices)  # type: ignore

    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        chat_completion = await self.client_async.chat.completions.create(stream=False, **self._chat_completion_assemble_params(messages, tool_definitions))
        return ChatCompletion(**chat_completion.dict())  # type: ignore

    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[ChatCompletionPart, None]:  # type: ignore
        chat_completion_stream = await self.client_async.chat.completions.create(stream=True, **self._chat_completion_assemble_params(messages, tool_definitions))
        async for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages: List[dict] = [c.delta.dict() for c in chunk.choices]  # type: ignore
                messages = [{**m, 'role': m.get('role') or 'assistant'} for m in messages]  # type: ignore
                messages = [{**m, 'content': m.get('content') or ''} for m in messages]  # type: ignore
                choices: List[dict] = [{'message': m} for m in messages]
                yield ChatCompletionPart(choices=choices)  # type: ignore

    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> TextCompletion:
        raise NotImplementedError()

    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Generator[TextCompletion, None, None]:
        raise NotImplementedError()

    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> Awaitable[TextCompletion]:
        raise NotImplementedError()

    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool] = []) -> AsyncGenerator[TextCompletion, None]:
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


def create_connector_llm(
    credentials: Credentials,
    hyperparameters: dict = {},
    provider: Literal['openai', 'azure_openai', 'llama_cpp'] = 'openai',
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
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Pseudo integration tests 1A - Simple messages - OpenAI
# this requires calls to openAI ($$$)
'''
llm = create_connector_llm(
    provider='azure_openai',
    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
    credentials=CredentialsOpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    ),
    hyperparameters = {'max_tokens': 3, 'tool_choice': None}
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
llm = create_connector_llm(
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


# Pseudo integration tests 2A - Tool calling - OpenAI
# this requires calls to openAI ($$$)
'''
llm = create_connector_llm(
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
