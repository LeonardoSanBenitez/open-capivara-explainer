import streamlit as st
import os
import time
import requests
from typing import List, Literal, Union, Generator

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
import json

# Sample data
# Should match what is on the capivara mock DB
alert_ids = ["dc2a", "8b66", "981a", "b3f7"]

def capivara_get_alert(id: str) -> Alert:
    '''
    Side effects: calls capivara
    Idempotent: yes
    '''
    response = requests.post('http://capivara:80/v1/public-api-read-incidents', json=[id])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])
    return Alert(**response.json()[0][1])

def capivara_update_conversation_customer(id: str, conversation: List[Message]) -> None:
    '''
    Side effects: calls capivara
    Idempotent: yes (updates the entire field in-place)
    '''
    response = requests.post('http://capivara:80/v1/public-api-update-incidents', json=[
        {
            "id": id,
            "conversation_customer": [m.dict() for m in conversation],
        }
    ])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])

def capivara_update_conversation_copilot(id: str, conversation: List[Message]) -> None:
    '''
    Side effects: calls capivara
    Idempotent: yes (updates the entire field in-place)
    '''
    response = requests.post('http://capivara:80/v1/public-api-update-incidents', json=[
        {
            "id": id,
            "conversation_copilot": [m.dict() for m in conversation],
        }
    ])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])

def generate_streamlit_writtables(message: Message) -> List[Union[str, dict]]:
    '''
    Side effects: none
    Idempotent: yes
    @param message: will be converted to writtables
    @return list of "writtables" (things you can pass to st.write).
    When returning pure text, the writtables are suitable for both streaming and entire responses.
    '''
    writtables = []
    if (message.intermediate_results is not None):
        for r in message.intermediate_results:
            writtables.append(f':gray[_At step {r.step}, the function `{r.function_name}` was called by the agent. Check details below._]')
            writtables.append({'Arguments': r.function_arguments, 'Result code': r.status_code, 'Result': r.result})
    if (message.message is not None) and (len(message.message) > 0):
            writtables.append(message.message)
    return writtables

def response_generator(model_copilot: Literal["Mock", "GPT3 agent", "GPT3", "Llama", "Llama agent"], selected_alert: Alert, question: str) -> Generator[Union[str, dict], None, None]:
    ####################################
    # Stream answer
    # Waits until get a full json
    # Supposes that the beginign of a json is always in the begining of a chunk
    response = requests.post(
        url='http://orchestrator-proxy:80/run_stream',
        json={'model_copilot': model_copilot, 'selected_alert': selected_alert.dict(), 'question': question},
        stream=True,
    )
    answer_parts: List[ValidatedAnswerPart] = []
    buffer = ""
    for chunk in response:
        if chunk:
            buffer += chunk.decode('utf-8')
            try:
                part = ValidatedAnswerPart(**json.loads(buffer))
                answer_parts.append(part)
                part_converted_to_message = Message(user='assistant', message=part.answer or '', intermediate_results=[ir.dict() for ir in part.intermediate_results or []])
                for writtable in generate_streamlit_writtables(message=part_converted_to_message):
                    yield writtable
                buffer = ""
            except json.JSONDecodeError:
                continue

    ####################################
    # Accumulate for saving in DB
    accumulated_answer = Message(user='assistant', message='', intermediate_results=[])
    for part in answer_parts:
        if part.answer is not None:
            accumulated_answer.message += part.answer
        if part.intermediate_results is not None:
            accumulated_answer.intermediate_results += [ir.dict() for ir in part.intermediate_results]
    selected_alert.conversation_copilot.append(accumulated_answer)
    capivara_update_conversation_copilot(selected_alert.id, selected_alert.conversation_copilot)


#########################################################
# Sidebar with alert IDs listed vertically
selected_alert_id = st.sidebar.radio("Select an Alert", alert_ids)

#########################################################
# Alert info
selected_alert: Alert = capivara_get_alert(selected_alert_id)
st.write("##### Title: " + selected_alert.title)
st.write(f"**Description:** {selected_alert.description}")
st.write(f"**Severity:** {selected_alert.severity}")
st.write(f"**State:** {selected_alert.state}\n\n---")
#st.write(f"DEBUG: messages {selected_alert.conversation_customer}")
#st.markdown("---")

conversation_customer, conversation_copilot = st.tabs(["Conversation with the customer", "Copilot"])

#########################################################
# Conversation with customer
with conversation_customer:
    container_customer = st.container(height=470)
    for message in selected_alert.conversation_customer:
        for writtable in generate_streamlit_writtables(message):
            container_customer.chat_message(message.user).write(writtable)

    prompt_customer = st.chat_input("Send a message to the customer")
    if prompt_customer:
        new_message = Message(user='user', message=prompt_customer)
        selected_alert.conversation_customer.append(new_message)
        capivara_update_conversation_customer(selected_alert_id, selected_alert.conversation_customer)
        for writtable in generate_streamlit_writtables(new_message):
            container_customer.chat_message(new_message.user).write(writtable)


#########################################################
# Conversation with copilot
with conversation_copilot:
    container_copilot = st.container(height=500)
    for message in selected_alert.conversation_copilot:
        for writtable in generate_streamlit_writtables(message):
            container_copilot.chat_message(message.user).write(writtable)
    

    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed
    with col1:
        prompt_copilot = st.chat_input("Chat with our copilot")
    with col2:
        model_copilot = st.selectbox('Select a model', ["GPT3 agent", "GPT3", "Llama", "Llama agent"], key="model_copilot", help='Which model to use', label_visibility='collapsed')
    if prompt_copilot:
        new_message = Message(user='user', message=prompt_copilot)
        selected_alert.conversation_copilot.append(new_message)
        capivara_update_conversation_copilot(selected_alert_id, selected_alert.conversation_copilot)
        for writtable in generate_streamlit_writtables(new_message):
            container_copilot.chat_message(new_message.user).write(writtable)

        # assistant response
        # The UI is updated inside write_stream, and the database is updated inside the generator
        response = container_copilot.chat_message("assistant").write_stream(response_generator(model_copilot, selected_alert, prompt_copilot))
