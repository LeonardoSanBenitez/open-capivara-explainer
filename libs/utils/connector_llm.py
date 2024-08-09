# type: ignore
from typing import Literal, List, Optional, AsyncGenerator, Generator, Awaitable
from pydantic import BaseModel, ConfigDict, computed_field
from functools import cached_property
from abc import ABC, abstractmethod
import openai
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
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
    tool_choice: str = "required"

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
    content: str
    role: str = 'assistant'


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
    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> ChatCompletion:
        pass

    @abstractmethod
    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> Generator[ChatCompletion, None, None]:
        pass

    @abstractmethod
    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> Awaitable[ChatCompletion]:
        pass

    @abstractmethod
    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> AsyncGenerator[ChatCompletion, None]:
        pass

    @abstractmethod
    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> TextCompletion:
        pass

    @abstractmethod
    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> Generator[TextCompletion, None, None]:
        pass

    @abstractmethod
    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> Awaitable[TextCompletion]:
        pass

    @abstractmethod
    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> AsyncGenerator[TextCompletion, None]:
        pass


class ConnectorLLMOpenAI(ConnectorLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    modelname: str
    hyperparameters: HyperparametersOpenAI
    credentials: CredentialsOpenAI

    @computed_field
    @cached_property
    def response_format(self) -> dict:
        if self.capabilities.response_json_only:
            return {"type": "json_object"}  # TODO: currently unused; hasto be set in hyperparam?
        else:
            return {"type": "text"}

    @computed_field
    @cached_property
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.credentials.base_url,
            api_key=self.credentials.api_key,
        )

    @computed_field
    @cached_property
    def client_async(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=self.credentials.base_url,
            api_key=self.credentials.api_key,
        )

    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> ChatCompletion:
        chat_completion = self.client.chat.completions.create(
            model=self.modelname,
            messages=[m.dict() for m in messages],
            tools=[d.dict() for d in tool_definitions],
            stream=False,
            **self.hyperparameters.dict(),
        )
        return ChatCompletion(**chat_completion.dict())

    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> Generator[ChatCompletionPart, None, None]:
        chat_completion_stream = self.client.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            tools=[d.dict() for d in tool_definitions],
            stream=True,
            **self.hyperparameters.dict(),
        )
        for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages_out: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages_out = [{**m, 'role': m.get('role') or 'assistant'} for m in messages_out]
                messages_out = [{**m, 'content': m.get('content') or ''} for m in messages_out]
                choices: List[dict] = [{'message': m} for m in messages_out]
                yield ChatCompletionPart(choices=choices)

    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> Awaitable[ChatCompletion]:
        chat_completion = await self.client_async.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            tools=[d.dict() for d in tool_definitions],
            stream=False,
            **self.hyperparameters.dict(),
        )
        return ChatCompletion(**chat_completion.dict())

    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool]) -> AsyncGenerator[ChatCompletionPart, None]:
        chat_completion_stream = await self.client_async.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            tools=[d.dict() for d in tool_definitions],
            stream=True,
            **self.hyperparameters.dict(),
        )
        async for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages = [{**m, 'role': m.get('role') or 'assistant'} for m in messages]
                messages = [{**m, 'content': m.get('content') or ''} for m in messages]
                choices: List[dict] = [{'message': m} for m in messages]
                yield ChatCompletionPart(choices=choices)

    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> TextCompletion:
        raise NotImplementedError()

    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> Generator[TextCompletion, None, None]:
        raise NotImplementedError()

    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> Awaitable[TextCompletion]:
        raise NotImplementedError()

    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiTool]) -> AsyncGenerator[TextCompletion, None]:
        raise NotImplementedError()


class ConnectorLLMAzureOpenAI(ConnectorLLMOpenAI):
    @computed_field
    @cached_property
    def client(self) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(
            azure_endpoint=self.credentials.base_url,
            api_key=self.credentials.api_key,
            api_version="2024-05-01-preview",
        )

    @computed_field
    @cached_property
    def client_async(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI(
            azure_endpoint=self.credentials.base_url,
            api_key=self.credentials.api_key,
            api_version="2024-05-01-preview",
        )


class ConnectorLlamaCPP(ConnectorLLM):
    pass


def create_connector_llm(
    credentials: Credentials,
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

        return ConnectorLLMOpenAI(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersOpenAI(),
            credentials=credentials,
        )
    elif provider == 'azure_openai':
        capabilities = Capabilities()
        if version == '1106' or version == '0125':
            capabilities.response_json_only = True

        return ConnectorLLMAzureOpenAI(
            modelname=modelname,
            capabilities=capabilities,
            hyperparameters=HyperparametersOpenAI(),
            credentials=credentials,
        )
    elif provider == 'llama_cpp':
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Pseudo tests 1 - Direct messages
# Does not really work anymore, since I'm forcing function call
# This can easily be refactored to make function call optional

'''
# this requires calls to openAI
#llm = create_connector_llm(
#    provider='azure_openai',
#    modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
#    version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
#    credentials=CredentialsOpenAI(
#        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
#        api_key=os.getenv("AZURE_OPENAI_KEY"),
#    ),
#)

llm = create_connector_llm(
    provider='openai',
    modelname='not-needed',
    version='not-needed',
    credentials=CredentialsOpenAI(
        base_url='http://llm-server:8080/v1',
        api_key='not-needed',
    ),
)


# text completion, normal
chat_completion = llm.chat_completion(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
chat_completion.choices[0].message.content


# chat completion, stream
chat_completion_stream = llm.chat_completion_stream(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message.content, end='')


# text completion, normal, async
chat_completion = await llm.chat_completion_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
chat_completion.choices[0].message.content


# chat completion, stream, async
chat_completion_stream = llm.chat_completion_stream_async(
    messages=[ChatCompletionMessage(content='hi', role='user')],
)
async for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message.content, end='')
'''


# Pseudo tests 2 - Function calls
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


# chat completion, stream, with function call
chat_completion_stream = llm.chat_completion_stream(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message, end='\n-----\n')


# chat completion, normal, async
chat_completion = await llm.chat_completion_async(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
print(chat_completion.choices[0].message.tool_calls[0].dict())


# chat completion, stream, async
chat_completion_stream = llm.chat_completion_stream_async(
    messages=[
        ChatCompletionMessage(**{'role': 'system', 'content': 'you are an AI assistant that answer only with function calls, nothing else. Always use the tools, do not answer without the tools.'}),
        ChatCompletionMessage(**{'role': 'user', 'content': 'What is the capital of brasil?'}),
    ],
    tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([(PluginCapital(), "PluginCapital")])),
)
async for chunk in chat_completion_stream:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].message, end='\n-----\n')
'''
