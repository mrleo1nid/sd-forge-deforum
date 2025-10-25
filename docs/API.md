# Deforum API Documentation

Complete REST API for programmatic access to Deforum video generation.

## Quick Start

### 1. Launch Forge with API Enabled

```bash
# Full API with job queue
python webui.py --deforum-api

# Simple API (safer, parameter whitelist)
python webui.py --deforum-simple-api

# Both APIs
python webui.py --deforum-api --deforum-simple-api
```

### 2. Access API Documentation

Once Forge is running:
- **Swagger UI:** http://localhost:7860/docs
- **ReDoc:** http://localhost:7860/redoc
- **OpenAPI JSON:** http://localhost:7860/openapi.json

### 3. Submit a Job

```bash
curl -X POST http://localhost:7860/deforum_api/batches \
  -H "Content-Type: application/json" \
  -d '{
    "deforum_settings": {
      "animation_mode": "3D",
      "max_frames": 120,
      "prompts": {"0": "a beautiful landscape"},
      "W": 512,
      "H": 512
    }
  }'
```

Response:
```json
{
  "message": "Job(s) accepted",
  "batch_id": "batch(123456789)",
  "job_ids": ["batch(123456789)-0"]
}
```

### 4. Check Job Status

```bash
curl http://localhost:7860/deforum_api/jobs/batch(123456789)-0
```

Response:
```json
{
  "id": "batch(123456789)-0",
  "status": "ACCEPTED",
  "phase": "GENERATING",
  "phase_progress": 0.45,
  "outdir": "/path/to/output",
  "timestring": "20250125043000"
}
```

## API Endpoints

### Batches

#### POST /deforum_api/batches
Submit a batch of video generation jobs.

**Request Body:**
```json
{
  "deforum_settings": { /* settings object or array */ },
  "options_overrides": { /* optional Forge settings */ }
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Job(s) accepted",
  "batch_id": "batch(123)",
  "job_ids": ["batch(123)-0", "batch(123)-1"]
}
```

---

#### GET /deforum_api/batches
List all batches and their job IDs.

**Response:** `200 OK`
```json
{
  "batch(123)": ["batch(123)-0", "batch(123)-1"],
  "batch(456)": ["batch(456)-0"]
}
```

---

#### GET /deforum_api/batches/{id}
Get detailed status of all jobs in a batch.

**Response:** `200 OK`
```json
[
  {
    "id": "batch(123)-0",
    "status": "SUCCEEDED",
    "phase": "DONE",
    "phase_progress": 1.0,
    ...
  }
]
```

---

#### DELETE /deforum_api/batches/{id}
Cancel all jobs in a batch.

**Response:** `200 OK`
```json
{
  "ids": ["batch(123)-0", "batch(123)-1"],
  "message": "2 job(s) cancelled."
}
```

### Jobs

#### GET /deforum_api/jobs
List all jobs across all batches.

**Response:** `200 OK`
```json
{
  "batch(123)-0": { /* DeforumJobStatus */ },
  "batch(456)-0": { /* DeforumJobStatus */ }
}
```

---

#### GET /deforum_api/jobs/{id}
Get detailed status of a specific job.

**Response:** `200 OK`
```json
{
  "id": "batch(123)-0",
  "status": "ACCEPTED",
  "phase": "GENERATING",
  "error_type": "NONE",
  "phase_progress": 0.45,
  "started_at": 1706150000.0,
  "last_updated": 1706150045.0,
  "execution_time": 45.0,
  "update_interval_time": 2.5,
  "updates": 18,
  "message": null,
  "outdir": "/path/to/output",
  "timestring": "20250125043000",
  "deforum_settings": { /* original settings */ },
  "options_overrides": null
}
```

---

#### DELETE /deforum_api/jobs/{id}
Cancel a specific job.

**Response:** `200 OK`
```json
{
  "id": "batch(123)-0",
  "message": "Job cancelled."
}
```

### Simple API

#### GET /deforum/api_version
Get Simple API version.

**Response:** `200 OK`
```json
{
  "version": "1.0"
}
```

---

#### GET /deforum/version
Get Deforum extension version.

**Response:** `200 OK`
```json
{
  "version": "6.5.0"
}
```

---

#### POST /deforum/run
Run generation with parameter whitelist (safer).

**Parameters:**
- `settings_json` (string): JSON settings object as string
- `allowed_params` (string): Semicolon-delimited parameter names

**Example:**
```bash
curl -X POST "http://localhost:7860/deforum/run" \
  -d "settings_json={\"max_frames\":120,\"prompts\":{\"0\":\"landscape\"}}" \
  -d "allowed_params=max_frames;prompts;W;H"
```

**Response:** `200 OK`
```json
{
  "outdir": "/path/to/output/run_id"
}
```

## Data Models

### DeforumJobStatus

```typescript
{
  id: string                      // Job identifier
  status: "ACCEPTED" | "SUCCEEDED" | "FAILED" | "CANCELLED"
  phase: "QUEUED" | "PREPARING" | "GENERATING" | "POST_PROCESSING" | "DONE"
  error_type: "NONE" | "RETRYABLE" | "TERMINAL"
  phase_progress: number          // 0.0 to 1.0
  started_at: number              // Unix timestamp
  last_updated: number            // Unix timestamp
  execution_time: number          // Seconds since start
  update_interval_time: number    // Seconds since last update
  updates: number                 // Update count
  message: string | null          // Error/status message
  outdir: string | null           // Output directory
  timestring: string | null       // Unique identifier for files
  deforum_settings: object | null // Original settings
  options_overrides: object | null // Forge overrides
}
```

### Job Lifecycle

```
QUEUED → PREPARING → GENERATING → POST_PROCESSING → DONE
  ↓          ↓            ↓              ↓
CANCELLED  FAILED       FAILED         FAILED
```

## Settings Format

Settings match the structure from `config/default_settings.txt`:

```json
{
  "animation_mode": "3D",
  "max_frames": 120,
  "prompts": {
    "0": "first prompt",
    "60": "second prompt"
  },
  "W": 512,
  "H": 512,
  "steps": 20,
  "sampler": "euler",
  "seed": -1,
  "fps": 15,
  // ... hundreds more parameters
}
```

**Tip:** Load `config/default_settings.txt` and modify only what you need.

## Error Responses

All endpoints return error responses in this format:

```json
{
  "id": "job_id",
  "status": "NOT FOUND",
  "message": "Job batch(123)-0 not found"
}
```

**Common Status Codes:**
- `202 Accepted` - Job queued successfully
- `400 Bad Request` - Invalid settings or parameters
- `404 Not Found` - Job/batch not found
- `500 Internal Server Error` - Generation failed

## Examples

### Python Client

```python
import requests
import time

API_URL = "http://localhost:7860/deforum_api"

# Submit batch
response = requests.post(f"{API_URL}/batches", json={
    "deforum_settings": {
        "animation_mode": "3D",
        "max_frames": 120,
        "prompts": {"0": "a beautiful landscape"},
        "W": 512,
        "H": 512
    }
})
job_id = response.json()["job_ids"][0]

# Poll status
while True:
    status = requests.get(f"{API_URL}/jobs/{job_id}").json()
    print(f"Progress: {status['phase_progress']:.1%}")

    if status['status'] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
        break

    time.sleep(5)

print(f"Output: {status['outdir']}")
```

### JavaScript Client

```javascript
const API_URL = 'http://localhost:7860/deforum_api';

// Submit batch
const response = await fetch(`${API_URL}/batches`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    deforum_settings: {
      animation_mode: '3D',
      max_frames: 120,
      prompts: {'0': 'a beautiful landscape'},
      W: 512,
      H: 512
    }
  })
});

const {job_ids} = await response.json();

// Poll status
const pollStatus = async () => {
  const status = await fetch(`${API_URL}/jobs/${job_ids[0]}`).then(r => r.json());
  console.log(`Progress: ${(status.phase_progress * 100).toFixed(1)}%`);

  if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(status.status)) {
    console.log(`Output: ${status.outdir}`);
    return;
  }

  setTimeout(pollStatus, 5000);
};

pollStatus();
```

## Best Practices

1. **Use minimal settings for testing:**
   ```json
   {
     "animation_mode": "3D",
     "max_frames": 3,
     "W": 512,
     "H": 512,
     "steps": 4,
     "seed": 42
   }
   ```
   This generates in ~30 seconds vs minutes for full settings.

2. **Poll status every 5 seconds, not continuously**

3. **Handle all job states:**
   ```python
   if status == 'SUCCEEDED':
       # Download video
   elif status == 'FAILED':
       # Log error message
   elif status == 'CANCELLED':
       # Clean up
   ```

4. **Set a timeout:**
   ```python
   start = time.time()
   while time.time() - start < 3600:  # 1 hour max
       # ... poll status
   ```

5. **Use the Simple API for untrusted input** - Parameter whitelist prevents malicious settings

## Concurrency

Currently limited to **1 concurrent job** via ThreadPoolExecutor. Jobs queue automatically.

Future: May support multiple concurrent jobs, but this requires careful memory management.

## Testing the API

### Quick Manual Test

Verify the API is working without running full tests:

```bash
# In the extension directory
python3 tests/manual_api_test.py
```

This script tests:
- ✓ Swagger UI is accessible at http://localhost:7860/docs
- ✓ OpenAPI schema is properly generated
- ✓ All endpoints are documented
- ✓ Response models are defined
- ✓ Basic endpoints work (list jobs/batches, version info)
- ✓ 404 responses for non-existent resources

### Swagger UI Interactive Testing

1. Open http://localhost:7860/docs in your browser
2. Expand any endpoint (e.g., "POST /deforum_api/batches")
3. Click "Try it out"
4. Edit the request body with your settings
5. Click "Execute"
6. See the response with status code and data

### Automated Tests

Run the comprehensive test suite:

```bash
# Run all API tests (includes GPU tests, slow)
cd ../../../  # Go to webui root
pytest extensions/sd-forge-deforum/tests/integration/ -v

# Run only fast Swagger tests (no GPU required)
pytest extensions/sd-forge-deforum/tests/integration/test_api_swagger.py -v
```

## See Also

- [MCP Integration](./MCP_INTEGRATION.md) - Use with Claude Desktop
- [Test Patterns](./GPU_TEST_PATTERNS.md) - Testing strategies
- [Swagger UI](http://localhost:7860/docs) - Interactive API explorer

## Troubleshooting

### Jobs stuck in QUEUED
- Check Forge console for errors
- Restart Forge with `--deforum-api`
- Cancel stuck jobs and resubmit

### Generation fails immediately
- Verify settings match default_settings.txt structure
- Check required fields: animation_mode, max_frames, prompts
- Test with minimal settings first

### Can't connect to API
- Ensure Forge launched with `--deforum-api` or `--deforum-simple-api`
- Check firewall allows localhost:7860
- Try http://localhost:7860/docs to verify API is running
