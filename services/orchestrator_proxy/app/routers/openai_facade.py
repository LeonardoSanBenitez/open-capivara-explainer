from typing import Optional, Tuple, List, Union, Any
from typing_extensions import Annotated
import re
import requests
import logging

from fastapi import APIRouter, Query, Request, Path, Header, Response
from fastapi.responses import StreamingResponse

from libs.plugin_orchestrator import rag
from libs.plugin_orchestrator.models import OpenaiCompletionResponse, EnumCitation

router = APIRouter()
# This set of APIs should be compatible with the OpenAI API
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
# Authentication: only "API Key authentication" is supported


async def stream_response(
    url: str,
    headers: dict,
    openai_key: str,
    body: dict,
    query_params: dict,
    extra_response: dict,
    citation_mode: EnumCitation,
):
    import openai
    response = openai.Completion.create(
       engine='gpt-35-turbo',
       prompt='what is the meaning of life?',
       temperature=1,
       max_tokens=100,
       top_p=0.5,
       frequency_penalty=0,
       presence_penalty=0,
       stop=None,
       stream=True
    )
    for chunk in response:
       yield json.dumps(chunk) + '\n'



async def normal_response(
    url: str,
    headers: dict,
    openai_key: str,
    body: dict,
    query_params: dict,
    extra_response: dict,
    citation_mode: EnumCitation,
) -> Tuple[int, dict]:
    '''
    return status_code, content
    '''
    # TODO: must be async
    response = requests.post(
        url,
        headers={
            'accept-encoding': headers.get('accept-encoding', ''),
            'accept': headers.get('accept', ''),
            'connection': headers.get('connection', ''),
            'api-key': openai_key,
            'content-type': headers.get('content-type', ''),
        },
        json=body,
        params=query_params,
    )
    # logging.info('>>>>>>>>>>?????>>>>>>>>>>> statys code', response.status_code)
    # logging.info('>>>>>>>>>>?????>>>>>>>>>>> response', response.json())
    content = response.json()
    if response.status_code // 100 == 2:
        content.update(extra_response)
    else:
        logging.warning(f'Request to openai failed: {response.status_code}, {content}')

    # TODO: implement citation mode
    # This is very similar to the implementation in `stream_response`

    return response.status_code, content


@router.post("/deployments/{deployment_id}/completions")
async def completions(
    *,
    request: Request,
    deployment_id: str,
    api_version: str = Query(default='2023-07-01-preview', alias="api-version"),
    response: Response,
    connection_string: Annotated[str, Header(description = 'API Key to connect to AzureOpenAI', alias="api-key")],  # noqa
) -> Union[Any, OpenaiCompletionResponse]:


    # Make actual request to OpenAI
    # TODO: how to "enrich response"? (e.g. returning document references or debug information)
    body = await request.json()
    if ('stream' in body) and (body['stream'] is True):
        return StreamingResponse(stream_response(
            body = body,
        ), media_type='text/event-stream')
    else:
        #status_code, content = await normal_response(
        #    url = f'{openai_protocol}://{openai_resource_name}.openai.azure.com/openai/deployments/{deployment_id}/completions',
        #    headers = headers,
        #    openai_key = openai_key,
        #    body = body,
        #    query_params = query_params,
        #    extra_response = extra_response,
        #    citation_mode = header_citation_mode,
        #)
        status_code, content = 200, {"message": "Hello World"}
        response.status_code = status_code
        return content


@router.post("/deployments/{deployment_id}/chat/completions")
async def chat_completions(
    *,
    request: Request,
    deployment_id: str,
    api_version: str = Query(default='2023-07-01-preview', alias="api-version"),
    response: Response,
    connection_string: Annotated[str, Header(description = 'API Key to connect to AzureOpenAI', alias="api-key")],  # noqa
):


    if ('stream' in body) and (body['stream'] is True):
        return StreamingResponse(stream_response(
            url = f'{openai_protocol}://{openai_resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions',
            headers = headers,
            openai_key = openai_key,
            body = body,
            query_params = query_params,
            extra_response = extra_response,
            citation_mode = header_citation_mode,
        ), media_type='text/event-stream')
    else:
        #status_code, content = await normal_response(
        #    url = f'{openai_protocol}://{openai_resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions',
        #    headers = headers,
        #    openai_key = openai_key,
        #    body = body,
        #    query_params = query_params,
        #    extra_response = extra_response,
        #    citation_mode = header_citation_mode,
        #)
        status_code, content = 200, {"message": "Hello World"}
        response.status_code = status_code
        return content
