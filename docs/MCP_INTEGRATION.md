# Model Context Protocol (MCP) Integration

Deforum provides an MCP server that allows Claude and other AI assistants to programmatically generate videos.

## What is MCP?

Model Context Protocol (MCP) is Anthropic's standardized protocol for connecting AI assistants to external tools and data sources. It allows Claude to:
- Call Deforum's API functions as tools
- Access Deforum settings templates as resources
- Use pre-built prompts for common tasks

## Features

### Tools
- `submit_deforum_batch` - Submit video generation jobs
- `get_job_status` - Check job progress and status
- `cancel_job` - Cancel running jobs
- `list_jobs` - List all jobs

### Resources
- `deforum://settings/default` - Full default settings template
- `deforum://settings/minimal` - Minimal settings for quick tests

### Prompts
- `deforum_simple_generation` - Quick video generation
- `deforum_flux_controlnet` - High-quality Flux + ControlNet generation

## Installation

### Prerequisites

1. **Install MCP package:**
```bash
pip install mcp
```

2. **Launch Forge with Deforum API:**
```bash
python webui.py --deforum-api
```

### Configure Claude Desktop

Add Deforum MCP server to your Claude Desktop configuration:

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "deforum": {
      "command": "python",
      "args": ["-m", "deforum.api.mcp_server"],
      "env": {
        "DEFORUM_API_URL": "http://localhost:7860/deforum_api"
      }
    }
  }
}
```

**Note:** Adjust the path if Forge runs on a different host/port.

## Usage Examples

### Example 1: Simple Generation

In Claude Desktop (with MCP configured):

```
Generate a 120-frame Deforum animation of "a serene mountain landscape at sunset"
```

Claude will:
1. Use the `submit_deforum_batch` tool
2. Submit a job with appropriate settings
3. Return the batch/job IDs
4. Optionally poll status with `get_job_status`

### Example 2: Check Job Status

```
Check the status of Deforum job batch(123456789)-0
```

Claude will call `get_job_status` and report:
- Current phase (QUEUED, GENERATING, etc.)
- Progress percentage
- Output directory
- Any error messages

### Example 3: Use a Prompt Template

```
Use the deforum_simple_generation prompt to create a video of "a cyberpunk city at night" with 60 frames
```

Claude will use the pre-built prompt template and execute the generation.

### Example 4: Advanced Settings

```
Generate a Deforum video with these specific settings:
- Animation mode: 3D
- Frames: 240
- Resolution: 768x768
- Prompt schedule:
  - Frame 0: "a seed growing"
  - Frame 60: "a small plant"
  - Frame 120: "a blooming flower"
  - Frame 180: "a fruit bearing tree"
```

Claude will construct the full settings object and submit it.

## API Endpoint Reference

When the MCP server makes requests, it uses these Deforum API endpoints:

- `POST /deforum_api/batches` - Submit batch
- `GET /deforum_api/jobs/{id}` - Get job status
- `DELETE /deforum_api/jobs/{id}` - Cancel job
- `GET /deforum_api/jobs` - List all jobs

See [Swagger docs](http://localhost:7860/docs) when Forge is running.

## Environment Variables

- `DEFORUM_API_URL` - Base URL for Deforum API (default: `http://localhost:7860/deforum_api`)

## Troubleshooting

### MCP server won't start
- Ensure `pip install mcp` is installed in Forge's venv
- Check that Claude Desktop config path is correct
- Verify JSON syntax in config file

### API connection errors
- Ensure Forge is running with `--deforum-api`
- Check `DEFORUM_API_URL` matches Forge's address
- Verify API is accessible: `curl http://localhost:7860/deforum_api/jobs`

### Jobs don't appear
- Check Forge console for errors
- Use Swagger UI to test API directly: http://localhost:7860/docs
- Verify settings format matches default_settings.txt structure

## Advanced Usage

### Custom MCP Server

You can extend the MCP server with additional tools:

```python
# In deforum/api/mcp_server.py

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        # ... existing tools ...
        Tool(
            name="my_custom_tool",
            description="My custom Deforum tool",
            inputSchema={...}
        )
    ]
```

### Standalone MCP Server

Run the MCP server standalone (without Claude Desktop):

```bash
python -m deforum.api.mcp_server
```

This starts an MCP server on stdio that can be used by any MCP client.

## See Also

- [Anthropic MCP Documentation](https://modelcontextprotocol.io/)
- [Deforum API Documentation](./API.md)
- [Deforum Settings Reference](../README.md)
