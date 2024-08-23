import streamlit as st
import json
import random
import time
import requests
from typing import List
from typing import Generator
from libs.plugin_orchestrator.answer_validation import ValidatedAnswerPart, IntermediateResult
import pandas as pd


# Sample data
alert_ids = ["dc2a", "981a", "b3f7"]

def capivara_get_alert(id: str) -> dict:
    response = requests.post('http://capivara:80/v1/public-api-read-incidents', json=[id])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])
    return response.json()[0][1]

def capivara_update_conversation_customer(id: str, conversation: list) -> None:
    response = requests.post('http://capivara:80/v1/public-api-update-incidents', json=[
        {
            "id": id,
            "conversation_customer": conversation,
        }
    ])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])

def capivara_update_conversation_copilot(id: str, conversation: list) -> None:
    response = requests.post('http://capivara:80/v1/public-api-update-incidents', json=[
        {
            "id": id,
            "conversation_copilot": conversation,
        }
    ])
    assert response.status_code == 200, f"Error: {response.text}"
    assert all([r[0] == 200 for r in response.json()])

def response_generator() -> Generator[str, None, None]:
    '''
    response = random.choice(
        [
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    '''

    responses: List[ValidatedAnswerPart] = [
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=1, status_code=200, function_name='PluginSAP_get_purchase_orders', function_arguments={}, result='[{"deliveryDate": "2021-01-01", "totalAmount": 10000, "items": [{"materialId": "IPS Natural Die Material ND1", "quantity": 1}, {"materialId": "Untersuchungshandschuhe Nitril light lemon Gr. M", "quantity": 10}, {"materialId": "Antiseptica r.f.u. H\\u00e4ndedesinfektion Flasche 1 Liter", "quantity": 1}]}]', message='At step 1, executing PluginSAP_get_purchase_orders', timestamp=1724422907, citation_id='d01b7fd7')]),
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=2, status_code=200, function_name='PluginSAP_create_purchase_order', function_arguments={'amount': 100, 'material_id': 'Untersuchungshandschuhe Nitril light lemon Gr. M'}, result='Order placed successfully', message='At step 2, executing PluginSAP_create_purchase_order', timestamp=1724422907, citation_id='ed7a943b')]),
        ValidatedAnswerPart(answer=None, citations=None, visualizations=None, intermediate_results=[IntermediateResult(step=3, status_code=200, function_name='PluginServiceNow_close_ticket', function_arguments={}, result='Ticket closed successfully', message='At step 3, executing PluginServiceNow_close_ticket', timestamp=1724422908, citation_id='64e0a3b0')]),
        ValidatedAnswerPart(answer='{"', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='text', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='":"', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='I', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' have', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' placed', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' an', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' order', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' for', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' ', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='100', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' units', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' of', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' the', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=" '", citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='Unt', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='ers', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='uch', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='ung', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='sh', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='ands', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='chu', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='he', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' N', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='itr', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='il', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' light', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' lemon', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' Gr', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='.', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' M', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer="'", citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' glove', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' model', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=',', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' as', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' requested', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='.', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' Additionally', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=',', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' I', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' have', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' closed', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' the', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' ticket', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' regarding', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' the', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' defective', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' product', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='.', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' Is', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' there', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' anything', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' else', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' I', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' can', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' assist', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' you', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer=' with', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='?"', citations=None, visualizations=None, intermediate_results=None),
        ValidatedAnswerPart(answer='}', citations=None, visualizations=None, intermediate_results=None),
    ]
    for response in responses:
        if response.answer is not None:
            yield response.answer
            #yield pd.DataFrame([['lala', {'mua': response.answer}]], columns=['name', 'args'])# {'lalala': response.answer}
        if (response.intermediate_results is not None) and (len(response.intermediate_results) > 0):
            r = response.intermediate_results[0]
            yield f':gray[_At step {r.step}, the function `{r.function_name}` was called by the agent. Check details below._]'
            yield {'Arguments': r.function_arguments, 'Result code': r.status_code, 'Result': r.result}
        time.sleep(0.05)

#########################################################
# Sidebar with alert IDs listed vertically
selected_alert_id = st.sidebar.radio("Select an Alert", alert_ids)

#########################################################
# Alert info
selected_alert = capivara_get_alert(selected_alert_id)
st.write("#### Title: " + selected_alert['title'])
st.write(f"**Description:** {selected_alert['description']}")
st.write(f"**Severity:** {selected_alert['severity']}")
st.write(f"**State:** {selected_alert.get('state', 'new')}")
st.write(f"DEBUG: messages {selected_alert['conversation_customer']}")
st.markdown("---")

conversation_customer, conversation_copilot = st.tabs(["Conversation with the customer", "Copilot"])

#########################################################
# Conversation with customer
with conversation_customer:
    container_customer = st.container(height=300)
    for message in selected_alert['conversation_customer']:
        container_customer.chat_message(message["user"]).write(message["message"])

    prompt_customer = st.chat_input("Send a message to the customer")
    if prompt_customer:
        selected_alert['conversation_customer'].append({"user": "user", "message": prompt_customer})
        capivara_update_conversation_customer(selected_alert_id, selected_alert['conversation_customer'])
        container_customer.chat_message("user").write(prompt_customer)


#########################################################
# Conversation with copilot
with conversation_copilot:
    container_copilot = st.container(height=300)
    for message in selected_alert['conversation_copilot']:
        container_copilot.chat_message(message["user"]).write(message["message"])
    
    prompt_copilot = st.chat_input("Chat with our copilot")
    if prompt_copilot:
        selected_alert['conversation_copilot'].append({"user": "user", "message": prompt_copilot})
        capivara_update_conversation_copilot(selected_alert_id, selected_alert['conversation_copilot'])
        container_copilot.chat_message("user").write(prompt_copilot)

        # assistant response
        response = container_copilot.chat_message("assistant").write_stream(response_generator())
        selected_alert['conversation_copilot'].append({"user": "assistant", "message": response})
        # TODO: the backend should be updated based on the original ValidatedAnswer, that contains all structured details
        #capivara_update_conversation_copilot(selected_alert_id, selected_alert['conversation_copilot'])
