import pandas as pd
import json
from typing import Tuple, Literal, List, Dict, Callable, Coroutine
from pydantic import BaseModel
from libs.utils.logger import get_logger
import pandas as pd
import os
import random

from libs.utils.connector_llm import ChatCompletionMessage
from libs.dataset_generator.exporter import DatasetExporterHuggingFace, DatasetExporterLocal
from libs.utils.prompt_manipulation import DefinitionOpenaiTool
from libs.plugins.plugin_capital import PluginCapital
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function
import libs.plugin_converter.openai_function_to_tool as openai_function_to_tool


logger = get_logger('libs.dataset_generator')


class DatasetGenerator(BaseModel):
    tool_definitions: List[DefinitionOpenaiTool]

    # This should return a system message that explains the functions that are available
    # The way the function are explained should coincide with the function calling syntax
    # TODO: some caller may want to use this modules to generate a dataset were the LLM does not receive a list of possible functions, instead if learn it... that case is not supported
    generate_system_message: Callable[[List[DefinitionOpenaiTool]], Coroutine[None, None, ChatCompletionMessage]]

    # To be used in case 1a and 1b
    generate_chitchat_trajectory: Callable[[], Coroutine[None, None, List[ChatCompletionMessage]]]

    # To be used in case 2a
    # dict maps toolsName -> function that generates a trajectory
    generate_sorry_trajectories: Dict[str, Callable[[], Coroutine[None, None, List[ChatCompletionMessage]]]]

    # To be used in case 2b
    # This should not contain the first system message (where the functions are explained); that is provided in `generate_system_message`
    # dict maps toolsName -> function that generates a trajectory
    generate_tool_trajectories: Dict[str, Callable[[], Coroutine[None, None, List[ChatCompletionMessage]]]]

    answer_function_name: str = 'answer'  # We suppose that this function is always available

    function_calling_syntax_source: Literal['capivara_orchestrator_bare', 'llm_compiler'] = 'capivara_orchestrator_bare'
    function_calling_syntax_target: Literal['capivara_orchestrator_bare', 'llm_compiler'] = 'capivara_orchestrator_bare'

    proportion_needs_tool: float = 0.9
    proportion_have_tool: float = 0.9
    n: int

    async def generate(self) -> List[List[ChatCompletionMessage]]:
        '''
        return a dataset
        Each item in the outer list represents a row
        Each row represents a trajectory

        A trajectory is simply a List[ChatCompletionMessage] containing a user instruction and the assistant responses (which probably calls tools)

        It will become a hugingface dataset... but it's not one... so maybe I should use another name

        -----

        The resulting dataset must cover all 4 cases:
        1a. the user does not need a tool (it's a normal conversation) and no tool was provided
        1b. the user does not need a tool but tools were provided
        2a. the user needs a tool but no tool was provided
        2b. the user needs a tool but and a suitable tool was provided

        Cases 1a and 1b are generated using the function generate_chitchat_trajectory
        Case 2a is generated using generate_sorry_trajectories
        Case 2b is generated using generate_tool_trajectories
        The proportions in the dataset are controlled by the parameters proportion_needs_tool and proportion_have_tool.

        TODO: for cases 1b and 2b, currently we pass all tools
        Maybe I should send a random subset of tools (for 2b that has to include the tool that is needed)
        '''
        # TODO: this method should be capable of converting between different tool calling syntaxes
        # The source (generated trajectories) can be in any of the supported syntaxes, whatever the caller prefers (since they are the unfortunate bastards that will have to write the code for it)
        # An alternative implementation should be to create ooone more syntax, a "generic" one, and obly the caller to use it. but nah
        # So conversion seems crumblesome, but it's the most flexible way to do it (and the most convinient for the caller)
        if self.function_calling_syntax_source != 'capivara_orchestrator_bare' and self.function_calling_syntax_target != 'capivara_orchestrator_bare':
            raise NotImplementedError('Only capivara_orchestrator_bare is supported')

        if not all([tool.function.name in self.generate_sorry_trajectories for tool in self.tool_definitions if tool.function.name != self.answer_function_name]):
            raise ValueError('Not all tools have a trajectory generator')
        if not all([tool.function.name in self.generate_tool_trajectories for tool in self.tool_definitions if tool.function.name != self.answer_function_name]):
            raise ValueError('Not all tools have a trajectory generator')

        dataset = []
        for i in range(self.n):
            trajectory: List[ChatCompletionMessage]
            if random.random() > self.proportion_needs_tool:
                # case 1: the user does not need a tool
                if random.random() > self.proportion_have_tool:
                    # case 1a: no tool was provided
                    _tool_definitions_active = list(filter(lambda d: d.function.name == self.answer_function_name, self.tool_definitions))
                    trajectory = [await self.generate_system_message(_tool_definitions_active)] + await self.generate_chitchat_trajectory()
                    logger.info(f"Generated trajectory {i} for case 1a")
                else:
                    # case 1b: tools were provided
                    # TODO: select random subset of tools, including 'answer'
                    _tool_definitions_active = self.tool_definitions
                    trajectory = [await self.generate_system_message(_tool_definitions_active)] + await self.generate_chitchat_trajectory()
                    logger.info(f"Generated trajectory {i} for case 1b")
            else:
                # case 2: the user needs a tool
                tool_name: str = random.choice([d.function.name for d in self.tool_definitions if d.function.name != self.answer_function_name])
                if random.random() > self.proportion_have_tool:
                    # case 2a: no tool was provided
                    _tool_definitions_active = list(filter(lambda d: d.function.name == self.answer_function_name, self.tool_definitions))
                    trajectory = [await self.generate_system_message(_tool_definitions_active)] + await self.generate_sorry_trajectories[tool_name]()
                    logger.info(f"Generated trajectory {i} for case 2a")
                else:
                    # case 2b: a suitable tool was provided
                    # TODO: select random subset of tools, including 'answer' and the one that is needed
                    _tool_definitions_active = self.tool_definitions
                    trajectory = [await self.generate_system_message(_tool_definitions_active)] + await self.generate_tool_trajectories[tool_name]()
                    logger.info(f"Generated trajectory {i} for case 2b")
            dataset.append(trajectory)
        assert len(dataset) == self.n
        assert all([type(row) == list for row in dataset])
        assert all([len(row) > 0 for row in dataset]), 'there are empty trajectories'
        assert all([all([type(message) == ChatCompletionMessage for message in row]) for row in dataset]), 'A trajectory should contain only ChatCompletionMessage objects'
        return dataset


#########
# Plugin specific code
from faker import Faker  # noqa
from libs.plugin_orchestrator.implementation_bare import OrchestratorBare  # noqa
from libs.plugins.plugin_capital import country_to_capital  # noqa


answer_function_name = 'answer'
faker = Faker()


async def generate_system_message(tool_definitions: List[DefinitionOpenaiTool]) -> ChatCompletionMessage:
    return ChatCompletionMessage(role='system', content=(
        f"You are allowed to call the following function:\n"
        f"{OrchestratorBare._prepare_prompt_function_description(tool_definitions)}\n"
    ))


async def generate_chitchat_trajectory() -> List[ChatCompletionMessage]:
    user_message = random.choice([
        'Hi',
    ])
    assistant_thought = 'I can answer the user.'
    assistant_answer = 'Hello, how are you doing?'
    return [
        ChatCompletionMessage(role='user', content=user_message),
        ChatCompletionMessage(role='assistant', content='{"thought": "' + assistant_thought + '", "action_name": "' + answer_function_name + '", "args": {"text": "' + assistant_answer + '"}}'),
    ]


async def generate_trajectory_sorry_PluginCapital_get_capital() -> List[ChatCompletionMessage]:
    # TODO: can I decrease the code duplication between this function and generate_trajectory_call_PluginCapital_get_capital?
    country = random.choice(list(country_to_capital.keys()))
    country_requested = random.choice([country, country.lower(), country.upper()])
    user_message = random.choice([
        f"What is the capital of {country_requested}?'",
        f"Can you tell me the capital of {country_requested}?'",
        f"I need help with a geography exam about {country_requested}, can you tell me what is its capital?'",
    ])
    assistant_thought = random.choice([
        f"I don't have the necessary tools to answer that'",
        f"I have no tools to get the necessary information'",
    ])
    assistant_answer = "Sorry, I don't know the answer to that question."
    return [
        ChatCompletionMessage(role='user', content=user_message),
        ChatCompletionMessage(role='assistant', content='{"thought": "' + assistant_thought + '", "action_name": "' + answer_function_name + '", "args": {"text": "' + assistant_answer + '"}}'),
    ]


async def generate_trajectory_call_PluginCapital_get_capital() -> List[ChatCompletionMessage]:
    '''
    Language: english
    Function calling syntax: capivara_orchestrator_bare
    Approximate number of unique trajectories:
    196 counties * 7 ways to ask the question * 4 ways to answer = 5488
    Plus each of those have differences in capitalization, punctuation, and thoughts, but that is minor
    '''
    country = random.choice(list(country_to_capital.keys()))
    country_requested = random.choice([country.lower(), country.upper(), country.capitalize()])
    user_message = random.choice([
        f"What is the capital of {country_requested}{random.choice(['?', ''])}",
        f"Can you tell me the capital of {country_requested}{random.choice(['?', ''])}",
        f"I need help with a geography exam about {country_requested}, can you tell me what is its capital?",
        # f'{faker.sentence()} What is the capital of {country_requested}?',
        f"My geography exam is tomorrow, {faker.date()}{random.choice(['.', ';'])} I'm sure the teacher will ask what is the capital of {country_requested}. Can you give me the correct answer now?",
        f"I have {faker.pyint(min_value=1, max_value=5)} minutes to find out what the capital of {country_requested} is. Can you answer that for me? Please{random.choice(['?', '!', ' :)', ''])}",
        f"I'm doing my school exam at {faker.url()}, and it asks what is the capital of {country_requested}. What is the answer?",
        f"My friends {faker.name()} and {faker.name()} said they are in the capital of {country_requested}. I was ashamed to ask what is the name of the city. Can you help me?",
    ])

    assistant_thought = random.choice([
        f"I need to retrieve the capital of {country}",
        f"I need to know the capital of {country}",
        f"This action will help me get the capital of {country}",
        f"The capital of {country} will be returned by PluginCapital_get_capital",
    ])
    return [
        ChatCompletionMessage(role='user', content=user_message),
        ChatCompletionMessage(role='assistant', content='{"thought": "' + assistant_thought + '", "action_name": "PluginCapital_get_capital", "args": {"country": "' + country + '"}}'),
    ]


generator = DatasetGenerator(
    tool_definitions=openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions([
        (PluginCapital(), "PluginCapital"),
    ])),
    generate_system_message=generate_system_message,
    generate_chitchat_trajectory=generate_chitchat_trajectory,
    generate_sorry_trajectories={
        'PluginCapital_get_capital': generate_trajectory_sorry_PluginCapital_get_capital,
    },
    generate_tool_trajectories={
        'PluginCapital_get_capital': generate_trajectory_call_PluginCapital_get_capital,
    },
    answer_function_name=answer_function_name,
    n=10000,
)

# end of plugin specific code
#########
