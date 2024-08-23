# Capivara Ticket Representation Standard

## Definitions
Ticket storage: exposes CRUD operations over stored tickets. Tickets follow CTRS syntax.

Endpoints: standarization of the APIs exposed by the ticket storage.

Ticket schema: structure of the JSON representing one ticket

Toolkit: series of utilities written in python to work with CTRS-compliant objects.

## Endpoints
* POST https://<domain>/v1/public-api-create-incidents
* POST https://<domain>/v1/public-api-read-incidents
* POST https://<domain>/v1/public-api-update-incidents
* POST https://<domain>/v1/public-api-delete-incidents
* POST https://<domain>/v1/availability-test

See `libs/CTRS/openapi.json` for a detailed definition of the endpoints.

## Ticket schema

See in "libs/CTRS/models.py"

## Toolkit

See in "libs/CTRS/*.py", plus examples in the notebooks "2.x - CTRS - ..."