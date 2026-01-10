from pydantic import BaseModel
from typing import Optional


class InputPayload(BaseModel):
    query: str
    context: Optional[str] = None
