# Plugin Orchestrator Evaluator
Module to evaluate plugin orchestrators (aka bots, agents, tool callers, etc).

This is very difficult to be super generic package, so it is more like a collection of useful scripts.

The metrics are (and should be) adjusted for each use case, specially the examples we provide to the judge LLM.

The metrics are defined in a jinja template, in GPT-style syntax. It is easy to trivial and convert to any message format you want (see `utils.parse_conversation` for an example).

# Known problems

for adversarial attacks, the prompt is sometimes blocked by the content filtering of the LLM provider.
That should be considered a refusal (currently it raises an error).