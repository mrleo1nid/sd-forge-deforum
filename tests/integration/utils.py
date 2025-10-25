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

from tenacity import retry, stop_after_delay, wait_fixed
import requests
from deforum.api.models import DeforumJobStatus, DeforumJobStatusCategory, DeforumJobPhase

SERVER_BASE_URL = "http://localhost:7860"
API_ROOT = "/deforum_api"
API_BASE_URL = SERVER_BASE_URL + API_ROOT

@retry(wait=wait_fixed(2), stop=stop_after_delay(900))
def wait_for_job_to_complete(id : str):
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    # Parse JSON manually instead of using PydanticSession
    # This gives better error messages if validation fails
    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    print(f"Waiting for job {id}: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
    assert jobStatus.status != DeforumJobStatusCategory.ACCEPTED
    return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_phase(id : str, phase : DeforumJobPhase):
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    print(f"Waiting for job {id} to enter phase {phase}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
    assert jobStatus.phase != phase
    return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_status(id : str, status : DeforumJobStatusCategory):
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    print(f"Waiting for job {id} to enter status {status}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
    assert jobStatus.status == status
    return jobStatus


def gpu_disabled():
    """Check if GPU is disabled on the server.

    Returns True if GPU is disabled, False if GPU is available.
    If the check fails (server error), assumes GPU is available (False).
    """
    try:
        response = requests.get(SERVER_BASE_URL+"/sdapi/v1/cmd-flags", timeout=5)
        response.raise_for_status()
        cmd_flags = response.json()
        return cmd_flags.get("use_cpu") == ["all"]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        # If we can't check GPU status, assume GPU is available
        # This prevents tests from being skipped due to server issues
        return False

        


