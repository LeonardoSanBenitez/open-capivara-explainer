generates a synthetic dataset appropriate for finetuning LLMs for function calling.

The dataset is specific to:
1. A toolset
2. A function calling syntax
3. Prompt format (suppots zephyr', 'chatml', 'llama2', 'llama3')

The third one is quite generic and easy to configure.  But ideally the first 2 you would be able to define the and then generate the dataset automatically, without writing code. However, we are not there yet, so you gotta write code that is completly locked-in to those two decisions, for every dataset you want to generate; so this module is more like a loose set of scripts than a standalone library.
We started working on making the 2nd configurable, but the 1st is hard.


Supports the resulting dataset to be saved to:
1. Local, as parquet
2. Uploaded to huggingface, as parquet

the split is done at export time, and it is not configurable beyond the proportions of train-test-split.


Supported prompt format (aka chat template or chat-over-text syntax):
* zephyr
* chatml (work in progress)
* llama2 (work in progress)
* llama3 (work in progress)

Function calling supported syntaxes:
* capivara_orchestrator_bare
* llm_compiler (work in progress)


