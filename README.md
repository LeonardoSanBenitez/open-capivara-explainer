# open-capivara-explainer

This is part of the OpenCapivara project, a not-yet opensource project developed by Skaylink.
You can find the other components of the project in the following repos:
* Main backend, https://github.com/skaylink/open-capivara
* Chat interface, https://github.com/skaylink/open-capivara-interface-chat
* Web interface, https://github.com/skaylink/open-capivara-interface-web

Since these repos are not yet opensource, all the interfaces to them will be mocked in the open-capivara-explainer repo.
Since the Capivara Ticket Representation Standard (CTRS) is described inside the open-capivara repo, it will be replicated here for reference.


## Getting started

The notebooks and container is for local dev only, the actual deployment is done with <TODO>.

Local file `.env` is required.

Run in the terminal, from the root folder:

```bash
docker-compose up
```

then go to `http://localhost:8888/tree?token=5074880b-def3-4506-be31-0f60c98cc42b` (or add this kernel in your VScode plugin)

See also the Spark UI at `http://localhost:4040/`. 