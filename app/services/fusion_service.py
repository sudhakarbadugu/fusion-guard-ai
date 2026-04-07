from app.models.schemas import IdentityResult, SceneResult, ActivityResult, FusionDecision

class FusionService:
    """
    FusionService implements the multimodal 'late fusion' logic for the system.
    It aggregates results from Identity, Scene, and Activity services to 
    produce a final security decision (RED, YELLOW, or GREEN alert).
    """
    def evaluate(self, 
                 identity: IdentityResult, 
                 scene: SceneResult, 
                 activity: ActivityResult) -> FusionDecision:
        """
        Combines multiple AI signals into a single actionable security decision.
        
        Logic Overview:
        - GREEN: Known person performing an authorized task.
        - YELLOW: Known person performing an unauthorized task (Policy violation).
        - RED: Unknown person or unidentified face (Security breach).
        
        Args:
            identity (IdentityResult): The output from IdentityService (WHO).
            scene (SceneResult): The output from SceneService (CONTEXT).
            activity (ActivityResult): The output from ActivityService (INTENT).
            
        Returns:
            FusionDecision: The final alert level and a descriptive message.
        """
        
        # Rule-based heuristic decision logic
        
        # CASE 1: Identity is verified and activity matches 'allowed' list.
        if identity.identity != "UNKNOWN" and activity.status == "AUTHORIZED":
            return FusionDecision(
                alert_level="GREEN",
                message=f"Safe: {identity.identity} recognized performing authorized activity: {activity.activity}."
            )
            
        # CASE 2: Identity is verified, but the activity is in the 'unauthorized' list.
        # This represents a known individual doing something they aren't supposed to.
        elif identity.identity != "UNKNOWN" and activity.status == "UNAUTHORIZED":
            return FusionDecision(
                alert_level="YELLOW",
                message=f"Warning: Known entity ({identity.identity}) performing unauthorized activity: {activity.activity}."
            )
            
        # CASE 3: Default fallback. 
        # If the face is UNKNOWN or NO_FACE_DETECTED, we trigger a high-level alert.
        else:
            return FusionDecision(
                alert_level="RED",
                message=f"Alert: Unknown or unauthorized entity detected. Scene: {scene.caption}."
            )
