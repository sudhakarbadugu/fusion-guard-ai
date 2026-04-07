from fastapi import APIRouter, File, UploadFile, Depends, Form
from typing import Optional, List
import numpy as np
import cv2
import json

from app.models.schemas import AnalysisResponse, PolicyRules
from app.services.identity_service import IdentityService
from app.services.scene_service import SceneService
from app.services.activity_service import ActivityService
from app.services.fusion_service import FusionService
from app.config import settings

router = APIRouter()

# Dependency injection
def get_identity_service(): return IdentityService()
def get_scene_service(): return SceneService()
def get_activity_service(): return ActivityService()
def get_fusion_service(): return FusionService()

@router.post("/enroll")
async def enroll_identity(
    identity: str = Form(...),
    file: UploadFile = File(...),
    identity_service: IdentityService = Depends(get_identity_service)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    success = identity_service.enroll(image, identity)
    if success:
        return {"status": "success", "message": f"Successfully enrolled identity: {identity}"}
    return {"status": "error", "message": "No face detected in the image"}

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_scene(
    file: UploadFile = File(...),
    allowed_activities: Optional[str] = Form(None),
    unauthorized_activities: Optional[str] = Form(None),
    identity_service: IdentityService = Depends(get_identity_service),
    scene_service: SceneService = Depends(get_scene_service),
    activity_service: ActivityService = Depends(get_activity_service),
    fusion_service: FusionService = Depends(get_fusion_service)
):
    # Process image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Decode JSON rules if provided
    allowed = settings.DEFAULT_ALLOWED_ACTIVITIES
    if allowed_activities and allowed_activities.strip():
        try:
            allowed = json.loads(allowed_activities)
        except json.JSONDecodeError:
            allowed = [x.strip() for x in allowed_activities.split(",")]

    unauthorized = settings.DEFAULT_UNAUTHORIZED_ACTIVITIES
    if unauthorized_activities and unauthorized_activities.strip():
        try:
            unauthorized = json.loads(unauthorized_activities)
        except json.JSONDecodeError:
            unauthorized = [x.strip() for x in unauthorized_activities.split(",")]

    # 1. Pipeline: Identity
    identity_result = identity_service.analyze(image)
    
    # 2. Pipeline: Scene
    scene_result = scene_service.analyze(image)
    
    # 3. Pipeline: Activity
    activity_result = activity_service.analyze(
        image=image,
        caption=scene_result.caption,
        allowed_activities=allowed,
        unauthorized_activities=unauthorized
    )
    
    # 4. Pipeline: Fusion
    decision_result = fusion_service.evaluate(
        identity=identity_result,
        scene=scene_result,
        activity=activity_result
    )

    return AnalysisResponse(
        identity=identity_result,
        scene=scene_result,
        activity=activity_result,
        decision=decision_result
    )
