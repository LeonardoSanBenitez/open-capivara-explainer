Supported libraries:
* [Azure OpenAI tool 2024-05-01-preview](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python-new#defining-functions)
* OpenAI tool (not tested, but should work)
* [Azure OpenAI function 2024-05-01-preview](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python-new#defining-functions)
* OpenAI function (not tested, but should work)
* OpenAI assistant (work in progress)
* semantic-kernel==0.5.0.dev0
* semantic-kernl==1.5.1 (work in progress)

Supported conversions:
* Semantic Kernel V0 -> OpenAI functions. Function `semantic_kernel_v0_to_openai_function.generate_definitions`.
* Semantic Kernel V0 -> OpenAI Assistant (work in progress)
* OpenAI functions -> OpenAI tool. Function `openai_function_to_tool.generate_definitions`
* OpenAI functions -> OpenAI Assistant  (work in progress)
* OpenAI Assistant -> OpenAI functions (work in progress)

Convertions are designed to be as lossless as possible, so you can chain, convert back-and-foward, etc


# Limitations
For library-specific limitations, see in the docstring of each conversion function

## Exception handling
If the function_to_be_wrapped raises an exception, the wrapper will raise an excpetion too

To increase the clarity for the LLM, the exceptions must have very specific classes:
* ValueError: for invalid parameters
* RuntimeError: for uncontrolable but somewhat excepted errors
Anything else will be considered an unexpected error, and its traceback will be logged

Ideally, we handle library-sepecific exceptions when making the conversion

# Sugar features

Things we added beyond the simple conversion

completly optional



## Built in plugins
Common plugins that can be automatically included in your application

These plugins are only the definition, your orchestrator have to implement the handling function.

If you are looking for full plugins (definiton + function), check if your orchestrator provide them.

Available plugins:
* answer_text (only the)
* answer_text_and_code
* answer_text_and_echarts
