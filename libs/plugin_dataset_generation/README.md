generates a synthetic dataset appropriate for finetuning LLMs for function calling.

The dataset is specific to:
1. A toolset
2. A function calling syntax

Ideally you would be able to define the 2 things above and then generate the dataset automatically, without writing code. However, we are not there yet, so you gotta write code that is completly locked-in to those two decisions, for every dataset you want to generate; so this module is more like a loose set of scripts than a standalone library.