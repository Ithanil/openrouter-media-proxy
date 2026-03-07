# OpenRouter Image Proxy

A minimal, transparent FastAPI proxy that lets **Open WebUI** (or any client
expecting the OpenAI Images API) generate and edit images via
**OpenRouter** models.

## Motivation

Open WebUI speaks the [OpenAI Images API][oai-images] — it calls
`/v1/images/generations` and `/v1/images/edits` with parameters like `prompt`,
`model`, `size`, `quality`, and `n`.

OpenRouter, however, exposes image generation through its **chat/completions**
endpoint using `modalities: ["image", "text"]` and returns base64 data-URLs
inside `message.images`.

This proxy sits between the two, translating on the fly:

```
Open WebUI ──► /v1/images/generations ──► Proxy ──► OpenRouter /chat/completions
                                                          │
Open WebUI ◄── { data: [{ b64_json }] } ◄── Proxy ◄──────┘
```

## What it does

| Incoming (OpenAI format)  | Outgoing (OpenRouter format)              |
|---------------------------|-------------------------------------------|
| `prompt`                  | `messages[0].content` (text or multimodal)|
| `model`                   | Passed through as-is to OpenRouter        |
| `size` (e.g. 1024x1536)  | `image_config.aspect_ratio` (e.g. 2:3)   |
| `quality` (hd, high, …)  | `image_config.image_size` (1K / 2K / 4K)  |
| `style` (vivid / natural) | Appended as a prompt hint                |
| `background: transparent` | Appended as a prompt hint                |
| `n` (number of images)    | Concurrent upstream requests              |
| `images` / `mask` (edits) | Inline `image_url` content parts          |

The model string is passed through to OpenRouter unchanged — configure
the desired OpenRouter model ID (e.g. `google/gemini-2.5-flash-image-preview`)
directly in Open WebUI's image generation settings.

The response is converted back into the standard
`{ created, data: [{ b64_json, revised_prompt }] }` format.

## Configuration

| Variable             | Default                                    | Description |
|----------------------|--------------------------------------------|-------------|
| `OPENROUTER_URL`     | `https://openrouter.ai/api/v1`             | OpenRouter API base URL. |
| `DEFAULT_MODALITIES` | `image,text`                               | Comma-separated. Use `image` for image-only models (Flux, Sourceful). |
| `UPSTREAM_TIMEOUT`   | `120`                                      | Request timeout in seconds. |
| `LOG_LEVEL`          | `INFO`                                     | Python logging level. |

## Quick Start

1. Place `openrouter_image_proxy.py`, `requirements.txt`, and
   `docker-compose.yml` in the same directory.
2. Start the service:
   ```bash
   docker compose up -d openrouter-image-proxy
   ```
3. Configure Open WebUI (Admin → Settings → Images):
   - **Image Generation Engine:** `OpenAI`
   - **API Base URL:** `http://openrouter-image-proxy:8080/v1`
   - **API Key:** your OpenRouter API key (`sk-or-v1-...`)
   - **Model:** an OpenRouter model ID with image output, e.g.
     `google/gemini-2.5-flash-image-preview`

## Usage Examples

### Generate an image

```bash
curl -s http://openrouter-image-proxy:8080/v1/images/generations \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy cabin in the mountains at sunset",
    "model": "google/gemini-2.5-flash-image-preview",
    "size": "1792x1024",
    "quality": "hd",
    "n": 1
  }' | jq '.data[0].b64_json' | head -c 80
```

### Edit an image

```bash
curl -s http://openrouter-image-proxy:8080/v1/images/edits \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [{"image_url": "data:image/png;base64,iVBORw0KGgo..."}],
    "prompt": "Add a rainbow in the sky",
    "model": "google/gemini-2.5-flash-image-preview",
    "size": "1024x1024"
  }'
```

## Limitations

- **Always returns `b64_json`.**  The `response_format: "url"` option is
  accepted but ignored — the proxy cannot host temporary image URLs.
- **No streaming.**  The `stream` / `partial_images` parameters are silently
  ignored.
- **`n > 1` = parallel requests.**  OpenRouter does not have an `n` parameter,
  so the proxy fires `n` concurrent upstream calls. This costs `n×` credits.
- **Masks are best-effort.**  The edits endpoint passes masks as additional
  image inputs; OpenRouter models may or may not interpret the mask as
  intended.
- **`file_id` references are not supported.**  Only `image_url` (including
  base64 data-URLs) works.
- **Style & background = prompt hints.**  `style: vivid|natural` and
  `background: transparent` are appended as natural-language instructions
  rather than structured parameters.

[oai-images]: https://platform.openai.com/docs/api-reference/images

