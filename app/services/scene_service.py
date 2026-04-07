import numpy as np
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from app.models.schemas import SceneResult

logger = logging.getLogger(__name__)

class SceneService:
    """
    SceneService generates natural language descriptions (captions) from input images.
    It utilizes the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce
    to perform image-to-text generation.
    """
    def __init__(self):
        """
        Initializes the BLIP processor and model.
        Automatically detects and utilizes the best available hardware (CUDA, MPS, or CPU).
        """
        logger.info("Initializing SceneService (BLIP-base)...")
        # Device priority: NVIDIA GPU (CUDA) > Apple Silicon (MPS) > Standard CPU
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load the base BLIP model which is optimized for standard image captioning tasks.
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def analyze(self, image: np.ndarray) -> SceneResult:
        """
        Generates a descriptive caption for a given image frame.
        
        Args:
            image (np.ndarray): The input image frame (BGR format from OpenCV).
            
        Returns:
            SceneResult: Encapsulated string caption describing the scene.
        """
        # Convert OpenCV image (BGR numpy array) to PIL Image (RGB) as expected by Transformers
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)

        # Preprocess the image and move tensors to the active device (GPU/CPU)
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        # Generate the caption using beam search or greedy decoding (default)
        # max_new_tokens=50 ensures the description stays concise but informative.
        out = self.model.generate(**inputs, max_new_tokens=500)
        
        # Decode the generated tokens back into a human-readable string
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        logger.info(f"Generated scene caption: {caption}")
        return SceneResult(caption=caption)
