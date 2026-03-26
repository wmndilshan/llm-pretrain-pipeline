"""
Inference Module

LLM inference services:
- Local server with WebSocket support
- Modal serverless deployment with auto-scaling
- Adaptive hardware detection
"""

from .server import (
    InferenceServer,
    GenerationConfig,
    HardwareDetector,
    ModelQuantizer,
)

# Modal inference (optional import)
try:
    from .modal_inference import (
        InferenceService,
        deploy_inference_service,
    )
except ImportError:
    # Modal not installed
    InferenceService = None
    deploy_inference_service = None

__all__ = [
    # Local server
    "InferenceServer",
    "GenerationConfig",
    "HardwareDetector",
    "ModelQuantizer",

    # Modal serverless
    "InferenceService",
    "deploy_inference_service",
]
