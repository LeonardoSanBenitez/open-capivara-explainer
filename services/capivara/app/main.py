from typing import List, Tuple, Optional
from fastapi import FastAPI
from libs.utils.logger import setup_loggers
from libs.CTRS.models import Alert, AlertUpdate, Message, IntermediateResult


setup_loggers()
app = FastAPI(debug=True)

mock_database: List[Alert] = [
    Alert( 
        id = "dc2a",
        title = "Defective product",
        description = "I ordered a Handstückschlauch, but it came already broken",
        severity = "Medium",
    ),
    Alert(
        id = "981a",
        title = "Help needed during installation",
        description = "I'm having trouble installing the Clean Reinigungsgerät, could you send someone to help me?",
        severity = "High",
        conversation_customer = [
            Message(user="customer", message="hello, any update?"),
        ],
    ),
    Alert(
        id = "b3f7",
        title = "Order more",
        description = "I want to buy more of the same gloves I bought last time, could you send 100 more?",
        severity = "Low",
        state = "closed",
        conversation_customer = [
            Message(user="user", message="I have placed the order as requested, you can follow the status in the order page."),
        ],
        conversation_copilot = [
            Message(user="user", message="What have this customer already bought?"),
            Message(
                user="assistant",
                intermediate_results=[IntermediateResult(step=1, status_code=200, function_name='PluginSAP_get_purchase_orders', function_arguments={}, result='[{"deliveryDate": "2021-01-01", "totalAmount": 10000, "items": [{"materialId": "IPS Natural Die Material ND1", "quantity": 1}, {"materialId": "Untersuchungshandschuhe Nitril light lemon Gr. M", "quantity": 10}, {"materialId": "Antiseptica r.f.u. H\\u00e4ndedesinfektion Flasche 1 Liter", "quantity": 1}]}]', message='At step 1, executing PluginSAP_get_purchase_orders', timestamp=1724422907, citation_id='d01b7fd7')],
                message="The customer have ordered the following items in the past: IPS Natural Die Material ND1, Untersuchungshandschuhe Nitril light lemon Gr. M, Antiseptica r.f.u. Händedesinfektion Flasche 1 Liter",
            ),
            Message(user="user", message="Buy 100 of those gloves"),
            Message(
                user="assistant",
                intermediate_results=[
                    IntermediateResult(step=1, status_code=200, function_name='PluginSAP_create_purchase_order', function_arguments={'amount': 100, 'material_id': 'Untersuchungshandschuhe Nitril light lemon Gr. M'}, result='Order placed successfully', message='At step 2, executing PluginSAP_create_purchase_order', timestamp=1724422907, citation_id='ed7a943b'),
                    IntermediateResult(step=2, status_code=200, function_name='PluginServiceNow_close_ticket', function_arguments={}, result='Ticket closed successfully', message='At step 3, executing PluginServiceNow_close_ticket', timestamp=1724422908, citation_id='64e0a3b0'),
                ],
                message="I have created a new order for 100 Untersuchungshandschuhe Nitril light lemon Gr. M, and closed this ticket for you 🙂",
            ),
        ],
    ),
]


@app.post("/v1/public-api-create-incidents")
async def create_incidents(
    incidents: List[Alert],
) -> List[Tuple[int, str]]:
    '''
    Returns tuples with status code (following HTTP status codes) and error message (if any).

    If successful, returns the incident id.
    '''
    mock_database.extend(incidents)
    return [(200, incident.id) for incident in incidents]


@app.post("/v1/public-api-read-incidents")
async def read_incidents(
    incident_ids: List[str],
) -> List[Tuple[int, Optional[Alert]]]:
    results: List[Tuple[int, Optional[Alert]]] = []
    for incident_id in incident_ids:
        incidents = list(filter(lambda x: x.id == incident_id, mock_database))
        if len(incidents) == 0:
            results.append((404, None))
        elif len(incidents) > 1:
            results.append((500, None))
        else:
            results.append((200, incidents[0]))
    return results


@app.post("/v1/public-api-update-incidents")
async def update_incidents(
    incident_updates: List[AlertUpdate],
) -> List[Tuple[int, Optional[str]]]:
    '''
    Partial updates for compose objects is not supported (if you pass a not none object, it is entirely updated)
    '''
    results: List[Tuple[int, Optional[str]]] = []
    for update in incident_updates:
        incidents = list(filter(lambda x: x.id == update.id, mock_database))
        if len(incidents) == 0:
            results.append((404, f"Incident with id {update.id} not found"))
        elif len(incidents) > 1:
            results.append((500, f"Multiple incidents with id {update.id} found"))
        else:
            incident = incidents[0]
            # Iterate over all the non-none fields and update the incident
            for field, value in update.dict().items():
                if value is not None:
                    setattr(incident, field, value)
            results.append((200, None))
    return results


@app.post("/v1/public-api-delete-incidents")
async def delete_incidents(
    incident_ids: List[str],
) -> List[Tuple[int, Optional[str]]]:
    '''
    Returns tuples with status code (following HTTP status codes) and error message (if any)
    '''
    global mock_database
    mock_database = list(filter(lambda x: x.id not in incident_ids, mock_database))
    return [(200, None) for _ in incident_ids]


@app.post("/v1/availability-test")
async def availability_test() -> int:
    return 200
