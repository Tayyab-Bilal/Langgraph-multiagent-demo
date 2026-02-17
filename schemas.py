from pydantic import BaseModel
from typing import Optional

class GreeterResponse(BaseModel):
    message: str
    intent: str
    reason: str
    email: str

class RetentionResponse(BaseModel):
    message: str
    outcome: str
    action: str

class ProcessorResponse(BaseModel):
    message: str

class SupportResponse(BaseModel):
    message: str
    resolved: bool