# FusionGuard AI - System Architecture & AI Assistant Guide

This document is intended to serve as a high-level reference for AI assistants (like Antigravity) and developers working on the `FusionGuard AI` codebase.

## 🚀 Project Overview

**Goal:** FusionGuard AI is a modular, multimodal machine learning pipeline designed for intelligent access control and context-aware security monitoring.
**Core Workflow (The Pipeline):**
1. **Identity** (InsightFace): Extracts face embeddings and matches them against known entities.
2. **Scene** (BLIP): Observes the image and safely extracts natural language context (Zero-shot VLM captioning).
3. **Activity** (CLIP): Evaluates the scene and/or image context against a dynamic list of approved/disallowed policy rules using Semantic representation.
4. **Fusion**: Merges the confidence thresholds of identity authorization and activity verification to output a final Red/Yellow/Green global threat indicator.

## 📁 Directory Structure & Key Files

- `app/main.py`: The FastAPI application entrypoint.
- `app/api/routes.py`: Contains API definitions (e.g. `/analyze` and `/enroll`). Dependency Injection is fully utilized here to map services into request lifecycles.
- `app/services/`: The core implementations of ML Logic.
    - `identity_service.py`: Hooks into PyTorch InsightFace (`buffalo_l`). Needs local embedding persistence in `data/embeddings`. 
    - `scene_service.py`: Implements Salesforce BLIP generic text captioning via HuggingFace `transformers`.
    - `activity_service.py`: Computes normalized similarities against policy rules using OpenAI CLIP embeddings.
    - `fusion_service.py`: Centralized rule processor managing late-fusion matrix evaluations based on individual service confidences.
- `app/config.py`: Contains `pydantic-settings`. Environmental overrides, thresholds (like `FACE_RECOGNITION_THRESHOLD`) and default policies.
- `app/models/schemas.py`: Pydantic object architectures defining API models (`IdentityResult`, `FusionDecision`, `AnalysisResponse`). 

## 🛠️ Environment & Commands

- **Language/Core**: Python 3.12+, FastAPI, PyTorch.
- **Run the API server**:
  ```bash
  source venv/bin/activate
  uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
  ```
- **Tests**:
  Use the `test_stubs.py` script constructed at project-root to trigger a dummy event without requiring Postman/Curl.
  ```bash
  python test_stubs.py
  ```

## 🧠 Model Initialization Details (Important for AI Agents)
- **Lazy Loading**: The ML models (BLIP, CLIP, InsightFace) are currently loaded lazily via FastAPI `Depends()` into memory. During debugging, ensure memory footprints are respected.
- **Hardware Acceleration**: Activity and Scene services look for CUDA -> MPS (Apple Silicon) -> CPU configurations upon initialization dynamically.
- **Thresholds**: If tests or model validations are not accurately capturing identity, recommend decreasing the `FACE_RECOGNITION_THRESHOLD` embedded into `app/config.py`.

## 📝 Modifying the System

**When adding a new feature pipeline (e.g. Audio Threat detection)**:
1. Define the input/output object inside `app/models/schemas.py`.
2. Construct `app/services/audio_service.py` to handle ML/Lib abstractions.
3. Update `app/services/fusion_service.py` to evaluate the new parameter in decision heuristics.
4. Inject the logic through FastAPI defaults in `app/api/routes.py`.
