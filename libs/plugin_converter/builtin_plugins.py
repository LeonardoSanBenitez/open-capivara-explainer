from pydantic import BaseModel
from libs.utils.prompt_manipulation import DefinitionOpenaiFunction, ParametersOpenaiFunction



answer_definition_text = DefinitionOpenaiFunction(
    name='answer',
    description="Send response back to the user. Show all your results here, this is the only thing that the user will see."
                " You won't be able to call any other function after this one."
                " Be sure to return a complete and clear answer, the user will not be able to see any other intermediate messages nor ask for more information."
                " Never mention intermediate messages or results; if you want to mention something, include it here."
                " Call this function only once, with everything you want to show to the user.",
    parameters=ParametersOpenaiFunction(
        type='object',
        properties={
            'text': {
                'type': 'string',
                'description': "final textual response to be send to the user. Should use HTML syntax for formatting."
            }
        },
        required=['text']
    ),
)

answer_definition_text_and_code = DefinitionOpenaiFunction(
    name='answer',
    description="Send response back to the user. Show all your results here, this is the only thing that the user will see."
                " You won't be able to call any other function after this one."
                " Be sure to return a complete and clear answer, the user will not be able to see any other intermediate messages nor ask for more information."
                " Never mention intermediate messages or results; if you want to mention something, include it here."
                " Call this function only once, with everything you want to show to the user.",
    parameters=ParametersOpenaiFunction(
        type='object',
        properties={
            'text': {
                'type': 'string',
                'description': "textual response to be sent to the user. Should use HTML syntax for formatting.",
            },
            'code': {
                'type': 'string',
                'description': "python code to generate visualization. Should always be valid code, that can be run as it is."
                               " If any variable is declared, use the `global` keyword, example: `global test; test = 10; print(10)`."
                               " Should be used only for visualization purposes, not for any other kind of code: any code snippet that you wish to show to the user should go in the 'text' parameter.",
            },
        },
        required=['text', 'code']
    ),
)

answer_definition_text_and_echart = DefinitionOpenaiFunction(
    name='answer',
    description="Send response back to the user. Show all your results here, this is the only thing that the user will see."
                " You won't be able to call any other function after this one."
                " Be sure to return a complete and clear answer, the user will not be able to see any other intermediate messages nor ask for more information."
                " Never mention intermediate messages or results; if you want to mention something, include it here."
                " Call this function only once, with everything you want to show to the user.",
    parameters=ParametersOpenaiFunction(
        type='object',
        properties={
            'text': {
                'type': 'string',
                'description': "textual response to be sent to the user. Should use HTML syntax for formatting."
                               " Do NOT include the HTML div to be used as placeholder for the graph, that is automatically added by the system.",
            },
            'echarts-definition': {
                'type': 'object',
                'description': "echarts `option`, to generate a valid and complete visualization."
                               " Should be used only for visualization purposes, not for any other kind of code or definition:"
                               " any code snippet that you wish to show to the user should go in the 'text' parameter.",
            },
        },
        required=['text', 'echarts-definition']
    ),
)
