import os
import numpy as np
import pickle
import logging
from insightface.app import FaceAnalysis
from app.models.schemas import IdentityResult
from app.config import settings
from numpy.linalg import norm

logger = logging.getLogger(__name__)

class IdentityService:
    """
    IdentityService handles face detection and recognition using the InsightFace library.
    It manages a local database of face embeddings (PKL files) and provides methods to 
    enroll new identities and analyze frames to identify known individuals.
    """
    def __init__(self):
        """
        Initializes the InsightFace FaceAnalysis application with the 'buffalo_l' model.
        Prepares the model for inference on the CPU and loads existing known embeddings.
        """
        logger.info("Initializing IdentityService (InsightFace)...")
        # Initialize InsightFace model
        # buffalo_l is a collection of models for detection, recognition, alignment, etc.
        # Using CPUExecutionProvider for broad compatibility across different environments.
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # Prepare the model: ctx_id=0 (GPU ID, -1 for CPU but InsightFace handles 0 for CPU if provider is CPU)
        # det_size=(640, 640) is the input resolution for the face detector.
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Directory where face embeddings (.pkl) are stored locally
        self.embeddings_dir = settings.EMBEDDINGS_DIR
        os.makedirs(self.embeddings_dir, exist_ok=True)
        # In-memory cache of identity:embedding mappings
        self.known_embeddings = self._load_known_embeddings()

    def _load_known_embeddings(self) -> dict:
        """
        Loads all stored face embeddings from the data directory into memory.
        
        Returns:
            dict: A dictionary mapping identity names to their 512-d feature vectors.
        """
        known = {}
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith(".pkl"):
                identity = os.path.splitext(filename)[0]
                with open(os.path.join(self.embeddings_dir, filename), "rb") as f:
                    known[identity] = pickle.load(f)
        return known

    def enroll(self, image: np.ndarray, identity: str) -> bool:
        """
        Extracts a face embedding from the provided image and saves it as a new identity.
        
        Args:
            image (np.ndarray): The input image (BGR format).
            identity (str): The unique name/ID for this person.
            
        Returns:
            bool: True if enrollment was successful (face found), False otherwise.
        """
        # Detect faces and extract features
        faces = self.app.get(image)
        if not faces:
            logger.warning(f"Enrollment failed: No face detected for {identity}")
            return False
        
        # Take the most prominent face (usually faces[0] in InsightFace sorted by detection score)
        embedding = faces[0].normed_embedding
        
        # Persist embedding to disk
        file_path = os.path.join(self.embeddings_dir, f"{identity}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(embedding, f)
            
        # Update in-memory cache
        self.known_embeddings[identity] = embedding
        logger.info(f"Successfully enrolled identity: {identity}")
        return True

    def analyze(self, image: np.ndarray) -> IdentityResult:
        """
        Detects the primary face in an image and compares it against enrolled identities.
        Uses cosine similarity between 512-dimensional embeddings.
        
        Args:
            image (np.ndarray): The input image to analyze.
            
        Returns:
            IdentityResult: Contains the matched identity name and the confidence score.
        """
        faces = self.app.get(image)
        if not faces:
            return IdentityResult(identity="NO_FACE_DETECTED", confidence=0.0)
            
        # Extract the embedding for the detected face
        target_embedding = faces[0].normed_embedding
        
        best_match = "UNKNOWN"
        best_score = 0.0
        
        # Compare target embedding with all known embeddings in the database
        for identity, known_embedding in self.known_embeddings.items():
            # InsightFace embeddings are normalized, so dot product equals cosine similarity.
            # Range is generally [-1, 1], with > 0.4 usually being a strong match for ArcFace.
            similarity = np.dot(target_embedding, known_embedding) 
            if similarity > best_score:
                best_score = similarity
                best_match = identity
        
        # Validate against configured threshold
        if best_score >= settings.FACE_RECOGNITION_THRESHOLD:
            return IdentityResult(identity=best_match, confidence=float(best_score))
            
        return IdentityResult(identity="UNKNOWN", confidence=float(best_score))
