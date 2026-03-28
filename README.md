# Asset Forge MCP

An MCP server that generates and edits game assets using the OpenAI Image API. Exposes tools over Streamable HTTP transport for use with Claude Code and other MCP clients.

## Features

- **generate_game_asset** — Create new game images (sprites, icons, portraits, backgrounds, tiles, UI elements)
- **edit_game_asset** — Edit existing images with text prompts and optional masks
- **generate_asset_variants** — Generate multiple variations of a concept in one call
- Inline base64 image delivery in MCP tool responses
- Persistent disk storage with sidecar metadata JSON
- Structured error responses with actionable messages
- Retry logic for rate limits and server errors

## Prerequisites

- Python 3.11+
- An OpenAI API key (or compatible image generation API)

## Installation

### With pip

```bash
pip install -e .
```

### With uv

```bash
uv pip install -e .
```

### Dev dependencies (for testing)

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Bearer token for the OpenAI API |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | API base URL (supports proxies) |
| `OPENAI_IMAGE_MODEL` | No | `gpt-image-1` | Image generation model |
| `ASSET_OUTPUT_DIR` | No | `./assets/generated` | Where generated assets are saved |
| `MCP_HOST` | No | `0.0.0.0` | Server bind host |
| `MCP_PORT` | No | `8080` | Server bind port |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

Environment variables take precedence over `.env` values.

## Running Locally

```bash
# Via entry point
asset-forge-mcp

# Or as a module
python -m asset_forge_mcp.server
```

The server starts on `http://0.0.0.0:8080` (or as configured).

## Docker

### Build

```bash
docker build -t asset-forge-mcp .
```

### Run (minimal, ephemeral assets)

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  asset-forge-mcp
```

Generated images are returned inline via MCP responses. Disk files inside the container are ephemeral.

### Run with persistent storage

```bash
docker run -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/assets:/app/assets/generated \
  asset-forge-mcp
```

### Docker Compose

```bash
docker compose up
```

The `docker-compose.yml` mounts `./assets` for persistent storage.

## Claude Code MCP Configuration

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "asset-forge": {
      "type": "streamable-http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

## Tool Usage Examples

### Generate a game asset

```
Generate a pixel-art sprite of a cute forest slime enemy for a top-down RPG.
Name it "forest_slime_idle" and tag it with "enemy", "forest", "slime".
```

Claude will call `generate_game_asset` with:
```json
{
  "name": "forest_slime_idle",
  "prompt": "cute forest slime enemy for a top-down RPG",
  "asset_type": "sprite",
  "style": "pixel-art",
  "background": "transparent",
  "tags": ["enemy", "forest", "slime"]
}
```

### Edit an existing asset

```
Take the forest slime and make it look more menacing while keeping the same silhouette.
```

Claude will call `edit_game_asset` with:
```json
{
  "input_path": "assets/generated/sprites/forest_slime_idle.png",
  "prompt": "make it more menacing while preserving the same silhouette",
  "output_name": "forest_slime_idle_menacing"
}
```

### Generate variants

```
Create 4 variants of a poison status icon for the RPG UI.
```

Claude will call `generate_asset_variants` with:
```json
{
  "name": "poison_icon",
  "prompt": "small poison status icon for fantasy RPG UI",
  "asset_type": "icon",
  "style": "vector",
  "variant_count": 4
}
```

## Inline Image Delivery

All tools return images as base64-encoded PNG content blocks directly in the MCP response. This means Claude can see and reason about the generated images immediately. Images are also saved to disk under `ASSET_OUTPUT_DIR` for persistent access.

Response structure:
1. A **text block** with JSON metadata (paths, asset type, tags, etc.)
2. One **image block** per generated image (base64 PNG)

## File Organization

Generated assets are organized by type:

```
assets/generated/
├── sprites/
├── icons/
├── portraits/
├── backgrounds/
├── tiles/
└── ui/
```

Each image gets a sidecar `.json` metadata file with the prompt, model, settings, and tags used to generate it.

## Running Tests

```bash
pytest tests/ -v
```

## Troubleshooting

**Server won't start: "OPENAI_API_KEY" validation error**
- Ensure `.env` exists with a valid API key, or pass it via environment variable.

**Connection refused from Claude Code**
- Verify the server is running and the port matches your MCP config.
- If running in Docker, ensure the port is mapped (`-p 8080:8080`).

**Images not persisting (Docker)**
- Mount a volume: `-v $(pwd)/assets:/app/assets/generated`

**Rate limit errors (429)**
- The server retries automatically (3 times with exponential backoff). If it still fails, wait and try again or check your OpenAI usage limits.

**Large response warnings**
- Generating many high-resolution variants may produce responses over 10 MB. The server logs a warning but still returns all images.
