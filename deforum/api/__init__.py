"""Deforum REST API module.

This module provides REST API endpoints for running Deforum generations
and managing job status through FastAPI integration.

API endpoints are automatically registered via script_callbacks.on_app_started()
when the module is imported (see bottom of api.py).
"""

from .api import JobStatusTracker
from .models import (
    Batch,
    DeforumJobErrorType,
    DeforumJobPhase,
    DeforumJobStatus,
    DeforumJobStatusCategory,
)

__all__ = [
    "JobStatusTracker",
    "Batch",
    "DeforumJobErrorType",
    "DeforumJobPhase",
    "DeforumJobStatus",
    "DeforumJobStatusCategory",
]
