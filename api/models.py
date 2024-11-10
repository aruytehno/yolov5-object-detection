from pydantic import BaseModel

class ScenarioState(BaseModel):
    action: str  # "start" or "stop"
