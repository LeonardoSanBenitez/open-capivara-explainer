import streamlit as st
import json
import random
import time
import requests
from typing import List, Union, Generator
from libs.plugin_orchestrator.answer_validation import ValidatedAnswerPart, IntermediateResult
from libs.CTRS.models import Alert, Message


# Sample data
# Should match what is on the capivara mock DB
alert_ids = ["dc2a", "981a", "b3f7"]

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

def response_generator(selected_alert: Alert) -> Generator[Union[str, dict], None, None]:
    answer_parts: List[ValidatedAnswerPart] = [
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=1, status_code=200, function_name='PluginSAP_get_purchase_orders', function_arguments={}, result='[{"deliveryDate": "2021-01-01", "totalAmount": 10000, "items": [{"materialId": "IPS Natural Die Material ND1", "quantity": 1}, {"materialId": "Untersuchungshandschuhe Nitril light lemon Gr. M", "quantity": 10}, {"materialId": "Antiseptica r.f.u. H\\u00e4ndedesinfektion Flasche 1 Liter", "quantity": 1}]}]', message='At step 1, executing PluginSAP_get_purchase_orders', timestamp=1724422907, citation_id='d01b7fd7')]),
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=2, status_code=200, function_name='PluginSAP_create_purchase_order', function_arguments={'amount': 100, 'material_id': 'Untersuchungshandschuhe Nitril light lemon Gr. M'}, result='Order placed successfully', message='At step 2, executing PluginSAP_create_purchase_order', timestamp=1724422907, citation_id='ed7a943b')]),
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=3, status_code=200, function_name='PluginServiceNow_close_ticket', function_arguments={}, result='Ticket closed successfully', message='At step 3, executing PluginServiceNow_close_ticket', timestamp=1724422908, citation_id='64e0a3b0')]),
        ValidatedAnswerPart(answer='{"text":"I have placed an orderv for ', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='100 units of the "Untersuchungshandschuhe Nitril light lemon Gr. M" glove model', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=', as requested. Additionally, I have closed the ticket regarding the defective product.', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' Is there anything else I can assist you with?"}', citations=None, visualizations=None, intermediate_results=None),
    ]
    
    # Stream answer
    for part in answer_parts:
        part_converted_to_message = Message(user='assistant', message=part.answer or '', intermediate_results=[ir.dict() for ir in part.intermediate_results or []])
        for writtable in generate_streamlit_writtables(message=part_converted_to_message):
            yield writtable
            time.sleep(0.05)

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
st.write("#### Title: " + selected_alert.title)
st.write(f"**Description:** {selected_alert.description}")
st.write(f"**Severity:** {selected_alert.severity}")
st.write(f"**State:** {selected_alert.state}")
#st.write(f"DEBUG: messages {selected_alert.conversation_customer}")
st.markdown("---")

conversation_customer, conversation_copilot = st.tabs(["Conversation with the customer", "Copilot"])

#########################################################
# Conversation with customer
with conversation_customer:
    container_customer = st.container(height=400)
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
    container_copilot = st.container(height=400)
    for message in selected_alert.conversation_copilot:
        for writtable in generate_streamlit_writtables(message):
            container_copilot.chat_message(message.user).write(writtable)
    
    prompt_copilot = st.chat_input("Chat with our copilot")
    if prompt_copilot:
        new_message = Message(user='user', message=prompt_copilot)
        selected_alert.conversation_copilot.append(new_message)
        capivara_update_conversation_copilot(selected_alert_id, selected_alert.conversation_copilot)
        for writtable in generate_streamlit_writtables(new_message):
            container_copilot.chat_message(new_message.user).write(writtable)

        # assistant response
        # The UI is updated inside write_stream, and the database is updated inside response_generator
        response = container_copilot.chat_message("assistant").write_stream(response_generator(selected_alert))

# TODO:
# when the orchestrator is validating the answer, it should remove `{"text":"` ... `")`
# Add selectbox to select which model to use. Should include gpt3, gpt3-agent, llama
# connec this with the actual agent, using GPT3