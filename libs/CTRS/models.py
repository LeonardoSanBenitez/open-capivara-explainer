from typing import Optional, Union
import uuid
from pydantic import BaseModel, Field
from typing import List, Literal
import time


'''
# About entities
they are a semi structured beast

have a `$id` field, integer, semi incremental (not always in order), starting at 1 (for sentinel alerts)

have a field `Type`. Its rough distribution is:
name                    | count     | example
------------------------|-----------|--------
account	                | 374       | {"$id":"3","Name":"floriano.peixoto","UPNSuffix":"exercito.com","IsDomainJoined":true,"DisplayName":"floriano.peixoto@exercito.com","Type":"account"}
url	                    | 234       | {"$id":"2","HostName":"desktop-9ir4nen","OSFamily":"Windows","OSVersion":"22H2","Type":"host","MdatpDeviceId":"94a845f157d61c50793c7988c955d5aef29ec5a7","FQDN":"desktop-9ir4nen","AadDeviceId":"0280296e-8c73-4f5c-bf61-58a4b5ad916b","RiskScore":"None","HealthStatus":"Active","LastSeen":"2023-05-31T06:06:48.2212819Z","LastExternalIpAddress":"46.5.72.186","LastIpAddress":"192.168.2.236","AvStatus":"NotSupported","OnboardingStatus":"Onboarded","LoggedOnUsers":[{"AccountName":"JürgenHaßlauer","DomainName":"AzureAD"}]}  # noqa
host	                | 212       |
process	                | 179       |
ip	                    | 164       |
file	                | 60        |
cloud-application	    | 39        |
mailbox	                | 27        |
mailMessage	            | 24        |
mailCluster	            | 21        |
azure-resource	        | 12        |
cloud-logon-session	    | 5         |
cloud-logon-request	    | 5         |
network-connection	    | 3         |
dns	                    | 3         |


can make references to each other, by using for example `{"$ref":"2"}` as value

'''


####################
class Signature(BaseModel):
    # All fields should be an array of floats
    pass


class SignatureV0(Signature):
    sentence_transformer_distiluse_base_multilingual_cased_v1: List[float]


class SignatureV1(Signature):
    sentence_transformers_clip_ViT_B32_multilingual_v1: List[float]
    is_outage: List[float]


class Signatures(BaseModel):
    # Key denotates the version/name of the signature
    # All keys should be named matching `type_versions`
    # All values should be of type Optional[Signature]
    v0: Optional[SignatureV0] = None
    v1: Optional[SignatureV1] = None


####################

class Decision(BaseModel):
    passed: bool = False  # Passed == True means that it is a FalsePositive
    classification: Optional[Literal['BenignPositive', 'FalsePositive', 'TruePositive', 'Undetermined']] = None
    details: str = ''


class CriteriaResult(BaseModel):
    criteria_name: str
    worked: bool
    passed: bool  # True = indicates a false positive
    weight: int
    details: str


class Message(BaseModel):
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    message: str
    user: str


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[Literal['High', 'Informational', 'Low', 'Medium']] = 'Low'
    category: Optional[str] = None
    state: Optional[Literal['new', 'analyzing', 'analysed', 'closed']] = 'new'
    version: Optional[str] = None
    source: Optional[str] = 'sentinel'

    decision: Decision = Field(default_factory=Decision)
    results: List[CriteriaResult] = []
    execution_time: Optional[float] = None
    signatures: Optional[Signatures] = None
    conversation_customer: List[Message] = []
    conversation_copilot: List[Message] = []

    workspace_id: Optional[str] = None
    timestamp: Optional[int] = None
    timestamp_orchestration: Optional[int] = None
    timestamp_generated: Optional[int] = None
    timestamp_closed: Optional[int] = None
    timestamp_first_activity: Optional[int] = None
    timestamp_last_activity: Optional[int] = None
    timestamp_first_modified: Optional[int] = None
    timestamp_last_modified: Optional[int] = None
    groups: List[str] = []
    alert_name: Optional[str] = None
    provider_name: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_original_id: Optional[str] = None
    resource_id: Optional[str] = None
    source_compute_id: Optional[str] = None
    alert_type: Optional[str] = None
    confidence_level: Optional[float] = None
    confidence_score: Optional[float] = None
    remediation_steps: Optional[str] = None
    extended_properties: dict = {}
    entities: List[dict] = []
    source_system: Optional[str] = None
    extended_links: list = []
    product_name: Optional[str] = None
    product_component_name: Optional[str] = None
    alert_link: Optional[str] = None
    compromised_entity: Optional[str] = None
    tactics: Optional[str] = None
    techniques: Optional[str] = None
    labels: List[str] = []

    classification: Optional[Literal['BenignPositive', 'FalsePositive', 'TruePositive', 'Undetermined']] = None
    classification_comment: Optional[str] = None
    classification_reason: Optional[Literal['InaccurateData', 'IncorrectAlertLogic', 'SuspiciousActivity', 'SuspiciousButExpected']] = None
    owner: Optional[dict] = None
    provider_incident_id: Optional[str] = None
    incident_number: Optional[int] = None
    related_analytic_rule_ids: list = []


class AlertUpdate(Alert):
    # Just turn everything optional
    groups: Optional[List[str]] = None  # type: ignore
    decision: Optional[Decision] = None  # type: ignore
    results: Optional[List[CriteriaResult]] = None  # type: ignore
    extended_properties: Optional[dict] = None  # type: ignore
    entities: Optional[List[dict]] = None  # type: ignore
    extended_links: Optional[list] = None  # type: ignore
    labels: Optional[List[str]] = None  # type: ignore
    related_analytic_rule_ids: Optional[list] = None  # type: ignore
