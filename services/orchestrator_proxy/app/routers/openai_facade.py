from typing import Optional, Tuple, List, Union, Any, AsyncGenerator
from typing_extensions import Annotated
import re
import requests
import logging

from fastapi import APIRouter, Query, Request, Path, Header, Response
from fastapi.responses import StreamingResponse


router = APIRouter()

async def stream_response_text_completion(body) -> AsyncGenerator[str, None]:
    response = requests.post(
        url='http://llm-server:8080/v1/completions',
        headers={'Content-Type': 'application/json'},
        json=body,
        stream=True,
    )
    for chunk in response:
        yield chunk.decode('utf-8')

async def stream_response_chat_completion(body) -> AsyncGenerator[str, None]:
    response = requests.post(
        url='http://llm-server:8080/v1/chat/completions',
        headers={'Content-Type': 'application/json'},
        json=body,
        stream=True,
    )
    for chunk in response:
        yield chunk.decode('utf-8')


@router.post("/completions")
async def completions(
    *,
    request: Request,
    api_version: str = Query(default='2023-07-01-preview', alias="api-version"),
    response: Response,
    connection_string: Annotated[str, Header(description = 'API Key to connect to AzureOpenAI', alias="Authorization")],  # noqa
):
    body = await request.json()
    if ('stream' in body) and (body['stream'] is True):
        return StreamingResponse(stream_response_text_completion(body), media_type='text/event-stream')
    else:
        return requests.post(
            url='http://llm-server:8080/v1/completions',
            headers={'Content-Type': 'application/json'},
            json=body,
        ).json()

@router.post("/chat/completions")
async def chat_completions(
    *,
    request: Request,
    api_version: str = Query(default='2023-07-01-preview', alias="api-version"),
    response: Response,
    connection_string: Annotated[str, Header(description = 'API Key to connect to AzureOpenAI', alias="Authorization")],  # noqa
):
    body = await request.json()
    if ('stream' in body) and (body['stream'] is True):
        return StreamingResponse(stream_response_chat_completion(body), media_type='text/event-stream')
    else:
        return requests.post(
            url='http://llm-server:8080/v1/chat/completions',
            headers={'Content-Type': 'application/json'},
            json=body,
        ).json()

