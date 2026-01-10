from __future__ import annotations
from typing import Dict, Optional, Literal
from pydantic import BaseModel


StageState = Literal["pending", "ok", "fail"]


class StageStatus(BaseModel):
    stage1: StageState = "pending"
    stage2: StageState = "pending"
    stage3: StageState = "pending"
    stage4: StageState = "pending"


class RunStatus(BaseModel):
    run_id: str
    stages: StageStatus = StageStatus()
    error: Optional[Dict[str, str]] = None
