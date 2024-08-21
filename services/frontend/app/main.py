import streamlit as st
import json
import random
import time
import requests
from typing import Generator

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
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
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
        capivara_update_conversation_copilot(selected_alert_id, selected_alert['conversation_copilot'])
