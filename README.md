# open-capivara-explainer

This is part of the OpenCapivara project, a not-yet opensource project developed by Skaylink.
You can find the other components of the project in the following repos:
* Main backend, https://github.com/skaylink/open-capivara
* Chat interface, https://github.com/skaylink/open-capivara-interface-chat
* Web interface, https://github.com/skaylink/open-capivara-interface-web

Since these repos are not yet opensource, all the interfaces to them will be mocked in the open-capivara-explainer repo.
Since the Capivara Ticket Representation Standard (CTRS) is described inside the open-capivara repo, it will be replicated here for reference.

## Components of the explainer

It was built as a set of loosely coupled moduled. Each module could have been published as a separate package, but we decided to include them all in this mono-repo.
We minimized the dependencies between them (except the `utils`: every calls things from utils, so it would have to be included in every module if they are published separately).
The modules are:
* plugin converter
* plugin orchestrator
* dataset generator


Additionally, this repo contains (these are not intended to be published as packages, never):
* a set of plugins for the capivara use case
* a very simple frontend with streamlit
* LLM server (which is just a local deployment of LlamaCPP)
* plugin orchestrator server (a mixture of library with deployment)
* LLM finetuner (set of scripts and best practices for finetuning)


## Getting started

The notebooks and container is for local dev only, the actual deployment is done with <TODO>.

Local file `.env` is required.

Run in the terminal, from the root folder:

```bash
docker-compose up
```

then go to `http://localhost:8888/tree?token=5074880b-def3-4506-be31-0f60c98cc42b` (or add this kernel in your VScode plugin)

See also the Spark UI at `http://localhost:4040/`. 