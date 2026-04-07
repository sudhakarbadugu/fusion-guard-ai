# GitHub Copilot Instructions for FusionGuard AI

This document provides context and instructions for GitHub Copilot when generating code, suggesting refactors, or writing tests for the **FusionGuard AI** repository.

## 🚀 Project Overview

**Goal:** FusionGuard AI is a modular, multimodal machine learning pipeline designed for intelligent access control and context-aware security monitoring.
**Core Workflow (The Pipeline):**
1. **Identity** (InsightFace): Extracts face embeddings and matches them against known entities (`app/services/identity_service.py`).
2. **Scene** (BLIP): Observes the image and extracts natural language context (`app/services/scene_service.py`).
3. **Activity** (CLIP): Evaluates the scene and/or image context against a dynamic list of approved/disallowed policy rules using Semantic representation (`app/services/activity_service.py`).
4. **Fusion**: Merges the confidence thresholds of identity authorization and activity verification to output a final Red/Yellow/Green global threat indicator (`app/services/fusion_service.py`).

## 🧠 Architectural Rules & Patterns

1. **Strict Typing:** All data payloads flowing through the API map directly to Pydantic objects located in `app/models/schemas.py`. Always use strict type hints.
2. **Dependency Injection:** The `app/api/routes.py` relies on FastAPI's `Depends()`. When creating new ML Services, always provide a dependency-injection wrapper and inject it into the route kwargs rather than instantiating the service inline.
3. **Lazy Loading:** All machine learning models (InsightFace, BLIP, CLIP) are loaded inside the `__init__` methods of their respective service classes. *Never load a model at the global module level!*
4. **Configuration:** Never hardcode ML thresholds (like minimum cosine similarity or API ports). Pull them from `app.config.settings`.

## 🛠️ Environment & Commands
- **Language/Core**: Python 3.12+, FastAPI, PyTorch.
- **Hardware Acceleration Check:** Models natively use strings like `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`. Ensure whenever you write new ML logic that PyTorch checks for these device cascades.

## 📝 Modifying the System

When asked to add a new pipeline section (e.g. Audio Model):
1. Define the resulting object in `app/models/schemas.py`.
2. Construct `app/services/audio_service.py` to handle ML/Lib abstractions.
3. Append logic inside `app/services/fusion_service.py` to evaluate the new parameter in decision heuristics.
4. Modify `AnalysisResponse` to return the resulting sub-element.
