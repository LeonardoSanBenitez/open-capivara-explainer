

from typing import Optional, Tuple, List, Union, Any, AsyncGenerator, Literal, Union, Generator
import os
import logging
import time
from flask import Flask, Response, stream_with_context, request

from libs.CTRS.models import Alert, Message
from libs.utils.connector_llm import factory_create_connector_llm, CredentialsOpenAI, ChatCompletionPart, ChatCompletionMessage
from libs.plugin_orchestrator.answer_validation import ValidatedAnswerPart, IntermediateResult
from libs.plugin_orchestrator.implementation_tool import OrchestratorWithTool
from libs.plugin_orchestrator.implementation_bare import OrchestratorBare
from libs.plugins.plugin_salesforce import PluginSalesforce
from libs.plugins.plugin_sap import PluginSAP
from libs.plugins.plugin_capivara import PluginCapivaraAlert
from libs.plugin_converter.semantic_kernel_v0_to_openai_function import generate_callables
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function
import libs.plugin_converter.openai_function_to_tool as openai_function_to_tool
from libs.utils.logger import setup_loggers


setup_loggers(logging.INFO)
app = Flask(__name__)

def stream_response_chat_completion(model_copilot: Literal["Mock", "GPT3 agent", "GPT3", "Llama", "Llama agent"], selected_alert: Alert, question: str) -> Generator[str, None, None]:
    base_system_prompt = (
        'You are an AI assistant whose goal is to help the use handle a support ticket.\n'
        'Here is some info about this ticket:\n'
        f'Title: {selected_alert.title}\n'
        f'Description: {selected_alert.description}\n'
        f'Severity: {selected_alert.severity}\n'
        'Answer the user questions and help the user to resolve this ticket.\n'
    )

    ####################################
    # Define agent/model
    answer_stream: Generator[Union[ValidatedAnswerPart, ChatCompletionPart], None, None]
    if model_copilot == 'Mock':  # or True:
        answer_stream = iter([
            ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=1, status_code=200, function_name='PluginSAP_get_purchase_orders', function_arguments={}, result='[{"deliveryDate": "2021-01-01", "totalAmount": 10000, "items": [{"materialId": "IPS Natural Die Material ND1", "quantity": 1}, {"materialId": "Untersuchungshandschuhe Nitril light lemon Gr. M", "quantity": 10}, {"materialId": "Antiseptica r.f.u. H\\u00e4ndedesinfektion Flasche 1 Liter", "quantity": 1}]}]', message='At step 1, executing PluginSAP_get_purchase_orders', timestamp=1724422907, citation_id='d01b7fd7')]),
            ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=2, status_code=200, function_name='PluginSAP_create_purchase_order', function_arguments={'amount': 100, 'material_id': 'Untersuchungshandschuhe Nitril light lemon Gr. M'}, result='Order placed successfully', message='At step 2, executing PluginSAP_create_purchase_order', timestamp=1724422907, citation_id='ed7a943b')]),
            ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=3, status_code=200, function_name='PluginServiceNow_close_ticket', function_arguments={}, result='Ticket closed successfully', message='At step 3, executing PluginServiceNow_close_ticket', timestamp=1724422908, citation_id='64e0a3b0')]),
            ValidatedAnswerPart(answer='I have placed an order for ', citations=None, visualizations=None, intermediate_results=None),
            ValidatedAnswerPart(answer='100 units of the "Untersuchungshandschuhe Nitril light lemon Gr. M" glove model', citations=None, visualizations=None, intermediate_results=None),
            ValidatedAnswerPart(answer=', as requested. Additionally, I have closed the ticket regarding the defective product.', citations=None, visualizations=None, intermediate_results=None),
            ValidatedAnswerPart(answer=' Is there anything else I can assist you with?', citations=None, visualizations=None, intermediate_results=None),
        ])
    elif 'agent' in model_copilot:
        # TODO: not only GPT3... probaly we can reuse a lot
        plugins = [
            (PluginCapivaraAlert(capivara_base_url='http://capivara:80', alert_id=selected_alert.id), "PluginServiceNow"),
            (PluginSalesforce(), "PluginSalesforce"),
            (PluginSAP(), "PluginSAP"),
        ]
        if model_copilot == 'GPT3 agent':
            llm = factory_create_connector_llm(
                provider='azure_openai',
                modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
                version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
                credentials=CredentialsOpenAI(
                    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                ),
            )

            orchestrator = OrchestratorWithTool(
                connection = llm,
                tool_definitions = openai_function_to_tool.generate_definitions(semantic_kernel_v0_to_openai_function.generate_definitions(plugins)),
                tool_callables = generate_callables(plugins),
                token_limit_input = 1024,
                token_limit_output = None,
                max_steps_recommended = 4,
                max_steps_allowed = 5,
                prompt_app_system=base_system_prompt,
                prompt_app_user=question,
            )
        elif model_copilot == 'Llama agent':
            # TODO
            raise NotImplementedError()
        else:
            raise RuntimeError(f'Invalid model: {model_copilot}')

        answer_stream = orchestrator.run_stream()
    else:
        # Non-agent chat
        # TODO: not only GPT3... probaly we can reuse a lot
        if model_copilot == 'GPT3':
            llm = factory_create_connector_llm(
                provider='azure_openai',
                modelname=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
                version=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT").split('-')[-1],
                credentials=CredentialsOpenAI(
                    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                ),
                hyperparameters = {'max_tokens': 1024, 'tool_choice': 'none'}
            )
        elif model_copilot == 'Llama':
            llm = factory_create_connector_llm(
                provider='llama_cpp',
                modelname='not-needed',
                version='not-needed',
                credentials=CredentialsOpenAI(
                    base_url='http://llm-server:8080/v1',
                    api_key='not-needed',
                ),
                hyperparameters = {'max_tokens': 1024}
            )
        else:
            raise RuntimeError(f'Invalid model: {model_copilot}')
        answer_stream = llm.chat_completion_stream(messages=[
            ChatCompletionMessage(role='system', content=base_system_prompt),
            ChatCompletionMessage(role='user', content=question),
        ])

    ####################################
    # Put the streaming in the expected Flask format
    def generate():
        for part in answer_stream:
            if isinstance(part, ChatCompletionPart):
                if (len(part.choices) > 0) and (part.choices[0].message is not None) and (part.choices[0].message.content is not None):
                    part = ValidatedAnswerPart(answer=part.choices[0].message.content)
                else:
                    continue
            assert isinstance(part, ValidatedAnswerPart)
            yield part.json()
            #time.sleep(0.5)
    
    return generate()

@app.route("/run_stream", methods=["POST"])
def chat_completions():
    data = request.get_json()
    assert type(data) == dict
    assert 'model_copilot' in data
    assert 'selected_alert' in data
    assert 'question' in data
    return Response(stream_response_chat_completion(model_copilot=data['model_copilot'], selected_alert=Alert(**data['selected_alert']), question=data['question']), content_type='application/json-seq')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
