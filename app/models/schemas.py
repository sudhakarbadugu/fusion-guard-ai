from pydantic import BaseModel, Field
from typing import List, Optional

class IdentityResult(BaseModel):
    identity: str = Field(description="Recognized identity or 'UNKNOWN'")
    confidence: float = Field(description="Confidence score of recognition")

class SceneResult(BaseModel):
    caption: str = Field(description="Generated description of the scene")

class ActivityResult(BaseModel):
    activity: str = Field(description="Closest matching activity description")
    status: str = Field(description="'AUTHORIZED' or 'UNAUTHORIZED'")
    confidence: float = Field(description="Confidence score of the classification")

class FusionDecision(BaseModel):
    alert_level: str = Field(description="Global threat level: 'GREEN', 'YELLOW', 'RED'")
    message: str = Field(description="Justification for the decision")

class AnalysisResponse(BaseModel):
    identity: IdentityResult
    scene: SceneResult
    activity: ActivityResult
    decision: FusionDecision

class PolicyRules(BaseModel):
    allowed_activities: Optional[List[str]] = None
    unauthorized_activities: Optional[List[str]] = None
