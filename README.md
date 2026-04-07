# FusionGuard-AI 🛡️

🔒 **Multimodal AI Security System | Face Recognition + Scene Understanding + Activity Verification using InsightFace, BLIP & CLIP with Late Fusion**

**FusionGuard-AI** is a modular, multimodal machine learning pipeline designed for intelligent access control and context-aware security monitoring.

Traditional security systems stop at biometric identification (e.g., face scans or keycards). FusionGuard goes a step further by leveraging Vision-Language Models (VLMs) to analyze the _context_ of a scene. By combining biometric identity verification with zero-shot activity classification, the system doesn't just ask "Who is this person?" but also evaluates "Are they authorized to perform this specific activity in this space?"

This decoupled architecture allows the system to be configured for any environment—from server rooms and corporate offices to restricted laboratories—simply by updating a configuration file of allowed and disallowed activities.

---

## 🧠 Core Architecture Pipeline

The system is built on a four-stage processing pipeline combining computer vision and natural language processing:

1. **Identity Verification (WHO):** Extracts face embeddings from the camera frame using InsightFace (ArcFace) to identify the individual against an enrolled local database (`app/services/identity_service.py`).
2. **Scene Understanding (WHAT):** Generates a natural language description (caption) of the ongoing activity using a Vision-Language Model like BLIP (`app/services/scene_service.py`).
3. **Contextual Authorization (IS IT ALLOWED):** Uses CLIP to perform zero-shot classification, matching the generated scene caption against a configurable, dynamic list of authorized and unauthorized behaviors (`app/services/activity_service.py`).
4. **Multimodal Fusion (DECISION):** Employs late fusion thresholding to evaluate the identity confidence score against the activity authorization status, generating a final security alert level: RED, YELLOW, or GREEN (`app/services/fusion_service.py`).

---

## 🚀 API Endpoints

The pipeline exposes a clean REST API built via **FastAPI**:
- `POST /api/v1/enroll`: Upload an image and provide an `identity` parameter to encode face embeddings locally in `data/embeddings/`.
- `POST /api/v1/analyze`: Upload a scene frame along with specific `allowed_activities` and `unauthorized_activities` to trigger the complete evaluation pipeline.

The service dynamically manages configuration overrides securely formatted against Pydantic definitions mapped in `app/models/schemas.py`.

---

## 🛠️ Environment Setup & Quickstart

### Prerequisites
- Python 3.12+ (Virtual Environment Recommended)
- Native system support for PyTorch (CUDA on Linux, MPS on Apple Silicon, or generic CPU mapping seamlessly handled).

### Installation
1. Clone the repository and navigate inside.
2. Initialize your Python virtual environment and install ML requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the Server
FastAPI applications are served natively through `uvicorn`:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
> **Note:** The underlying ML models (BLIP, CLIP, InsightFace) are loaded lazily upon the *first* request. Ensure you have high-bandwidth network connectivity during the first run as it pulls approximately ~4-5GB of weights down to your global HuggingFace cache!

### Using Interactive Docs
Once the server is running, navigate to:
**http://127.0.0.1:8000/docs** to interface automatically utilizing Swagger's interactive payload templates.

## 🧪 Testing

To sanity test the integration of all services without Postman:
```bash
python test_stubs.py
```
This script dynamically executes a local payload ping leveraging `FastAPI.TestClient` entirely in-memory using dummy numpy blobs.

---
## ✨ Built With
- **FastAPI / Pydantic** - Clean architecture logic wrapping.
- **HuggingFace Transformers** - Orchestrating BLIP / CLIP context models.
- **InsightFace** - Performing state-of-the-art ArcFace encoding routines.
- **PyTorch** / **OpenCV** - Native computational graphs.
