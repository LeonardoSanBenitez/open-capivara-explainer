This is very difficult to be super generic package

the metrics are (and should be) adjusted for each use case, specially the examples we provide to the judge LLM


----

also, the supported LLM format is currently... skaylink format? capivara orchestrator? skaylink prompt-flow based?

the metrics are defined in a jinja template, in GPT-style syntax. It is easy to trivial and convert to any message format you want (see `utils.parse_conversation` for an example).