from typing import List, Tuple, Optional
from fastapi import FastAPI
from libs.utils.logger import setup_loggers
from libs.CTRS.models import Alert, AlertUpdate


setup_loggers()
app = FastAPI(debug=True)

mock_database: List[Alert] = []


@app.post("/v1/public-api-create-incidents")
async def create_incidents(
    incidents: List[Alert],
) -> List[Tuple[int, Optional[str]]]:
    '''
    Returns tuples with status code (following HTTP status codes) and error message (if any)
    '''
    mock_database.extend(incidents)
    return [(200, None) for _ in incidents]


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
