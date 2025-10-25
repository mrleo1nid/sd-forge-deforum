"""Model Context Protocol (MCP) Server for Deforum.

Exposes Deforum functionality via MCP for use by Claude and other AI assistants.
Allows programmatic access to Deforum's video generation capabilities.

Launch with: python -m deforum.api.mcp_server
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, Resource, Prompt, TextContent, ImageContent
    import mcp.server.stdio
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    raise

# Will be imported when running within Forge
import requests

# API base URL - can be configured via environment variable
import os
API_BASE_URL = os.getenv("DEFORUM_API_URL", "http://localhost:7860/deforum_api")


# Initialize MCP server
app = Server("deforum")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Deforum tools."""
    return [
        Tool(
            name="submit_deforum_batch",
            description="Submit a Deforum video generation batch job. "
                        "Generates an animated video based on the provided settings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "deforum_settings": {
                        "type": "object",
                        "description": "Deforum settings object (matches settings.txt format)",
                        "properties": {
                            "animation_mode": {
                                "type": "string",
                                "enum": ["3D", "Interpolation"],
                                "description": "Animation mode (3D for depth-warped, Interpolation for morphing)"
                            },
                            "max_frames": {
                                "type": "integer",
                                "description": "Total number of frames to generate"
                            },
                            "prompts": {
                                "type": "object",
                                "description": "Frame-indexed prompts, e.g., {'0': 'first prompt', '60': 'second prompt'}"
                            },
                            "W": {"type": "integer", "description": "Width in pixels"},
                            "H": {"type": "integer", "description": "Height in pixels"},
                        },
                        "required": ["animation_mode", "max_frames", "prompts"]
                    },
                    "options_overrides": {
                        "type": "object",
                        "description": "Optional Forge settings to override"
                    }
                },
                "required": ["deforum_settings"]
            }
        ),
        Tool(
            name="get_job_status",
            description="Get the current status of a Deforum generation job. "
                        "Returns progress, phase, and output information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID returned from submit_deforum_batch"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="cancel_job",
            description="Cancel a running or queued Deforum job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to cancel"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="list_jobs",
            description="List all Deforum jobs with their current status.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""

    if name == "submit_deforum_batch":
        # Submit batch to Deforum API
        deforum_settings = arguments.get("deforum_settings", {})
        options_overrides = arguments.get("options_overrides")

        payload = {"deforum_settings": deforum_settings}
        if options_overrides:
            payload["options_overrides"] = options_overrides

        response = requests.post(f"{API_BASE_URL}/batches", json=payload)
        response.raise_for_status()
        result = response.json()

        return [TextContent(
            type="text",
            text=f"Batch submitted successfully!\n"
                 f"Batch ID: {result['batch_id']}\n"
                 f"Job IDs: {', '.join(result['job_ids'])}\n\n"
                 f"Use get_job_status with job ID to check progress."
        )]

    elif name == "get_job_status":
        job_id = arguments["job_id"]
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        status = response.json()

        # Format status nicely
        text = f"Job Status: {status['status']}\n"
        text += f"Phase: {status['phase']}\n"
        text += f"Progress: {status['phase_progress']:.1%}\n"
        text += f"Execution time: {status['execution_time']:.1f}s\n"

        if status.get('message'):
            text += f"Message: {status['message']}\n"

        if status.get('outdir'):
            text += f"\nOutput directory: {status['outdir']}\n"
            if status.get('timestring'):
                text += f"Timestring: {status['timestring']}\n"

        return [TextContent(type="text", text=text)]

    elif name == "cancel_job":
        job_id = arguments["job_id"]
        response = requests.delete(f"{API_BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        result = response.json()

        return [TextContent(
            type="text",
            text=f"Job {job_id} cancelled: {result['message']}"
        )]

    elif name == "list_jobs":
        response = requests.get(f"{API_BASE_URL}/jobs")
        response.raise_for_status()
        jobs = response.json()

        if not jobs:
            return [TextContent(type="text", text="No jobs found.")]

        # Format jobs list
        text = f"Total jobs: {len(jobs)}\n\n"
        for job_id, status in jobs.items():
            text += f"Job {job_id}:\n"
            text += f"  Status: {status['status']}\n"
            text += f"  Phase: {status['phase']}\n"
            text += f"  Progress: {status['phase_progress']:.1%}\n"
            text += "\n"

        return [TextContent(type="text", text=text)]

    else:
        raise ValueError(f"Unknown tool: {name}")


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="deforum://settings/default",
            name="Default Deforum Settings",
            mimeType="application/json",
            description="Default settings template for Deforum generation"
        ),
        Resource(
            uri="deforum://settings/minimal",
            name="Minimal Deforum Settings",
            mimeType="application/json",
            description="Minimal settings for fast test generation (3 frames, 512x512)"
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content."""

    if uri == "deforum://settings/default":
        # Load default settings
        settings_path = Path(__file__).parent.parent / "config" / "default_settings.txt"
        with open(settings_path, 'r', encoding='utf-8') as f:
            return f.read()

    elif uri == "deforum://settings/minimal":
        # Return minimal settings for testing
        minimal = {
            "animation_mode": "3D",
            "max_frames": 3,
            "prompts": {"0": "a beautiful landscape"},
            "W": 512,
            "H": 512,
            "steps": 4,
            "sampler": "euler",
            "seed": 42
        }
        return json.dumps(minimal, indent=2)

    else:
        raise ValueError(f"Unknown resource: {uri}")


@app.list_prompts()
async def list_prompts() -> List[Prompt]:
    """List available prompt templates."""
    return [
        Prompt(
            name="deforum_simple_generation",
            description="Generate a simple Deforum animation with minimal settings",
            arguments=[
                {
                    "name": "description",
                    "description": "What to generate (e.g., 'a beautiful landscape')",
                    "required": True
                },
                {
                    "name": "frames",
                    "description": "Number of frames (default: 120)",
                    "required": False
                }
            ]
        ),
        Prompt(
            name="deforum_flux_controlnet",
            description="Generate with Flux model and ControlNet depth control",
            arguments=[
                {
                    "name": "description",
                    "description": "What to generate",
                    "required": True
                }
            ]
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: Dict[str, str]) -> str:
    """Get prompt content."""

    if name == "deforum_simple_generation":
        description = arguments.get("description", "a beautiful landscape")
        frames = arguments.get("frames", "120")

        return f"""Generate a Deforum animation with these settings:

Use the submit_deforum_batch tool with:
{{
    "deforum_settings": {{
        "animation_mode": "3D",
        "max_frames": {frames},
        "prompts": {{"0": "{description}"}},
        "W": 512,
        "H": 512
    }}
}}

This will create a {frames}-frame animation of "{description}".
"""

    elif name == "deforum_flux_controlnet":
        description = arguments.get("description", "a beautiful landscape")

        return f"""Generate a Flux ControlNet animation with:

Use the submit_deforum_batch tool with:
{{
    "deforum_settings": {{
        "animation_mode": "3D",
        "max_frames": 120,
        "prompts": {{"0": "{description}"}},
        "enable_flux_controlnet_v2": true,
        "flux_controlnet_strength": 0.8,
        "flux_controlnet_model": "depth",
        "W": 512,
        "H": 512
    }}
}}

This will use Flux with ControlNet depth guidance for higher quality.
"""

    else:
        raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
