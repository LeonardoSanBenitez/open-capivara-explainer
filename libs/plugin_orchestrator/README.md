Multiple independent but interchangable implementations, for different underlying capabilities of the LLM endpoint.

Available implementations:
* tool-based (for endpoints that support explicit tool calling, like OpenAI; see [docs](https://platform.openai.com/docs/guides/function-calling))
* assistant-based (for endpoints that support explicit tool calling, like OpenAI; see [docs](https://platform.openai.com/docs/assistants/overview))
* bare (for endpoints that don't support any function-related mechanism) (work in progress)



# Sugar features

## Answer validation

Before calling your tool, we check if the LLM answer satisfy a definition.

For implementation that 

Available built in answer validators:
* text in HTML + code of language "python-matplotlib" + echarts definition + citation (default)

specially useful if used together with an "answer" tool that induces the LLM to generate an structured output

for having structured multi-ouput answers (aka structured outputs), you should define your own 'answer' function and pass it as the `answer_definition` parameter of the `generate_definitions_from_<lib>_plugins` function

==maybe this should be moved to the PluginOrchestrator?==