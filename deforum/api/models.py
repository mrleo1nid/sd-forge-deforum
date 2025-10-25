# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class Batch(BaseModel):
    """Request model for submitting a batch of Deforum generation jobs.

    A batch can contain one or multiple Deforum settings objects.
    Each settings object represents a separate video generation job.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "deforum_settings": {
                    "animation_mode": "3D",
                    "max_frames": 120,
                    "prompts": {"0": "a beautiful landscape"},
                    "W": 512,
                    "H": 512
                },
                "options_overrides": {
                    "deforum_save_gen_info_as_srt": True
                }
            }
        }
    )

    deforum_settings: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Single settings object or list of settings objects. "
                    "Each object should match the structure from a saved Deforum settings.txt file."
    )
    options_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Forge settings to override for this batch (e.g., save_gen_info_as_srt)"
    )

class DeforumJobStatusCategory(str, Enum):
    """Overall status of a Deforum job."""
    ACCEPTED = "ACCEPTED"      # Job accepted and queued
    SUCCEEDED = "SUCCEEDED"    # Job completed successfully
    FAILED = "FAILED"          # Job failed with error
    CANCELLED = "CANCELLED"    # Job cancelled by user


class DeforumJobPhase(str, Enum):
    """Current phase of job execution."""
    QUEUED = "QUEUED"                      # Waiting in queue
    PREPARING = "PREPARING"                # Loading models, preparing args
    GENERATING = "GENERATING"              # Generating frames
    POST_PROCESSING = "POST_PROCESSING"    # Video assembly, upscaling, etc.
    DONE = "DONE"                          # All phases complete


class DeforumJobErrorType(str, Enum):
    """Type of error if job failed."""
    NONE = "NONE"              # No error
    RETRYABLE = "RETRYABLE"    # Temporary error, retry may succeed
    TERMINAL = "TERMINAL"      # Permanent error, retry won't help

class DeforumJobStatus(BaseModel):
    """Detailed status information for a Deforum generation job.

    Tracks the job's progress through various phases and provides
    timing information and output locations.
    """
    model_config = ConfigDict(frozen=True, extra='ignore')

    id: str = Field(description="Unique job identifier")
    status: DeforumJobStatusCategory = Field(description="Overall job status")
    phase: DeforumJobPhase = Field(description="Current execution phase")
    error_type: DeforumJobErrorType = Field(description="Error type if job failed")
    phase_progress: float = Field(
        description="Progress within current phase (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    started_at: float = Field(description="Timestamp when job started (Unix epoch)")
    last_updated: float = Field(description="Timestamp of last status update (Unix epoch)")
    execution_time: float = Field(description="Time elapsed since job start (seconds)")
    update_interval_time: float = Field(description="Time since last status update (seconds)")
    updates: int = Field(description="Number of status updates received", ge=0)
    message: Optional[str] = Field(default=None, description="Error message or status details")
    outdir: Optional[str] = Field(default=None, description="Output directory for generated files")
    timestring: Optional[str] = Field(default=None, description="Unique timestring identifier for output files")
    deforum_settings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Original Deforum settings used for this job"
    )
    options_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Forge options overrides used for this job"
    )


class BatchSubmitResponse(BaseModel):
    """Response when a batch is successfully submitted."""
    message: str = Field(description="Success message")
    batch_id: str = Field(description="Unique batch identifier")
    job_ids: List[str] = Field(description="List of job IDs in this batch")


class BatchCancelResponse(BaseModel):
    """Response when cancelling a batch."""
    ids: List[str] = Field(description="List of cancelled job IDs")
    message: str = Field(description="Cancellation result message")


class JobCancelResponse(BaseModel):
    """Response when cancelling a single job."""
    id: str = Field(description="Cancelled job ID")
    message: str = Field(description="Cancellation result message")


class ErrorResponse(BaseModel):
    """Standard error response."""
    id: Optional[str] = Field(default=None, description="Job or batch ID if applicable")
    status: Optional[str] = Field(default=None, description="Status code or category")
    message: str = Field(description="Error message")


class VersionResponse(BaseModel):
    """API or Deforum version information."""
    version: str = Field(description="Version string")


class SimpleRunResponse(BaseModel):
    """Response from the simple /deforum/run endpoint."""
    outdir: str = Field(description="Output directory where generated files will be saved")