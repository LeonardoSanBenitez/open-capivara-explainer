# type: ignore
from typing import Literal, List, Optional, AsyncGenerator, Generator, Awaitable
from pydantic import BaseModel, ConfigDict, computed_field
from functools import cached_property
from abc import ABC, abstractmethod
import openai
from libs.utils.prompt_manipulation import DefinitionOpenaiFunction
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


class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage


class ChatCompletion(BaseModel):
    choices: List[ChatCompletionChoice]

###


class ConnectorLLM(BaseModel, ABC):
    capabilities: Capabilities
    hyperparameters: Hyperparameters
    credentials: Credentials

    @abstractmethod
    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> ChatCompletion:
        pass

    @abstractmethod
    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> Generator[ChatCompletion, None, None]:
        pass

    @abstractmethod
    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> Awaitable[ChatCompletion]:
        pass

    @abstractmethod
    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> AsyncGenerator[ChatCompletion, None]:
        pass

    @abstractmethod
    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> TextCompletion:
        pass

    @abstractmethod
    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> Generator[TextCompletion, None, None]:
        pass

    @abstractmethod
    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> Awaitable[TextCompletion]:
        pass

    @abstractmethod
    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> AsyncGenerator[TextCompletion, None]:
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

    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> ChatCompletion:
        chat_completion = self.client.chat.completions.create(
            model=self.modelname,
            messages=[m.dict() for m in messages],
            functions=[d.dict() for d in tool_definitions],
            # tool_choice="required",
            stream=False,
            **self.hyperparameters.dict(),
        )
        print(chat_completion)
        return ChatCompletion(**chat_completion.dict())

    def chat_completion_stream(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> Generator[ChatCompletion, None, None]:
        chat_completion_stream = self.client.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            stream=True,
            **self.hyperparameters.dict(),
        )
        for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages_out: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages_out = [{**m, 'role': m.get('role') or 'assistant'} for m in messages_out]
                messages_out = [{**m, 'content': m.get('content') or ''} for m in messages_out]
                choices: List[dict] = [{'message': m} for m in messages_out]
                yield ChatCompletion(choices=choices)

    async def chat_completion_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> Awaitable[ChatCompletion]:
        chat_completion = await self.client_async.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            stream=False,
            **self.hyperparameters.dict(),
        )
        return ChatCompletion(**chat_completion.dict())

    async def chat_completion_stream_async(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiFunction]) -> AsyncGenerator[ChatCompletion, None]:
        chat_completion_stream = await self.client_async.chat.completions.create(
            model=self.modelname,
            messages=[m.dict()for m in messages],
            stream=True,
            **self.hyperparameters.dict(),
        )
        async for chunk in chat_completion_stream:
            if len(chunk.choices) > 0:
                messages: List[dict] = [c.delta.dict() for c in chunk.choices]
                messages = [{**m, 'role': m.get('role') or 'assistant'} for m in messages]
                messages = [{**m, 'content': m.get('content') or ''} for m in messages]
                choices: List[dict] = [{'message': m} for m in messages]
                yield ChatCompletion(choices=choices)

    def text_completion(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> TextCompletion:
        raise NotImplementedError()

    def text_completion_stream(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> Generator[TextCompletion, None, None]:
        raise NotImplementedError()

    async def text_completion_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> Awaitable[TextCompletion]:
        raise NotImplementedError()

    async def text_completion_stream_async(self, prompt: str, tool_definitions: List[DefinitionOpenaiFunction]) -> AsyncGenerator[TextCompletion, None]:
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


# Pseudo tests

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
