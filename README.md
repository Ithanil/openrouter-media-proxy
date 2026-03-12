# OpenRouter Media Proxy

A minimal, transparent FastAPI proxy that lets **Open WebUI** or any other
client expecting parts of the **OpenAI Images API** and **OpenAI Audio API**
talk to **OpenRouter** models exposed through `/chat/completions`.

## Motivation

Open WebUI expects OpenAI-style endpoints such as:

- `/v1/images/generations`
- `/v1/images/edits`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/audio/speech`

OpenRouter exposes both image generation and audio input/output through
`/api/v1/chat/completions`, using multimodal `messages`, `input_audio`, image
parts, and streamed audio deltas.

This proxy sits in between and translates on the fly:

```text
Open WebUI / OpenAI client
    -> /v1/images/*, /v1/audio/*
    -> Proxy
    -> OpenRouter /chat/completions
```

## Supported Endpoints

### Images

- `POST /v1/images/generations`
- `POST /v1/images/edits`

Image requests are translated into OpenRouter `modalities: ["image", "text"]`
style chat requests. Returned base64 data URLs are converted back into OpenAI
`{ created, data: [{ b64_json, revised_prompt? }] }` responses.

### Audio

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`

Audio transcription and translation requests accept multipart uploads,
base64-encode the file, and send it to OpenRouter as an `input_audio` content
part. Speech requests translate OpenAI TTS-style JSON into OpenRouter audio
output requests and collect the upstream SSE audio chunks back into a binary
audio response.

Bare `/images/*` and `/audio/*` routes are also available for convenience.

## Request Mapping

### Images

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| `prompt` | `messages[0].content` |
| `model` | Passed through unchanged |
| `size` | `image_config.aspect_ratio` |
| `quality` | `image_config.image_size` |
| `style` / `background` | Prompt hints |
| `n` | Parallel upstream requests |
| uploaded images / masks | inline `image_url` parts |

### Audio Input

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| multipart `file` | `input_audio.data` base64 |
| file type / extension | `input_audio.format` |
| `model` | Passed through unchanged |
| `prompt` / `language` | Instruction text |
| `response_format` | Instruction shaping + response normalization |
| `temperature` | Passed through when present |

### Audio Output

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| `input` | user message content |
| `model` | Passed through unchanged |
| `voice` | `audio.voice` |
| `response_format` | `audio.format` |
| `instructions` / `speed` | system prompt hints |
| `stream_format: "audio"` | proxy collects SSE and returns audio bytes |
| `stream_format: "sse"` | proxy returns a simple SSE stream of audio chunks |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `DEFAULT_MODALITIES` | `image,text` | Legacy image-only env var kept for compatibility. |
| `DEFAULT_IMAGE_MODALITIES` | `image,text` | Preferred env var for image-capable models. |
| `UPSTREAM_TIMEOUT` | `120` | Request timeout in seconds. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Quick Start

1. Start the service:

   ```bash
   docker compose up -d openrouter-media-proxy
   ```

2. Point your client at:

   ```text
   http://openrouter-media-proxy:8080/v1
   ```

3. Use an OpenRouter API key in the normal OpenAI `Authorization: Bearer ...`
   header.

4. Configure model IDs in the client as **OpenRouter model IDs**, not native
   OpenAI IDs.

## Usage Examples

### Generate an image

```bash
curl -s http://openrouter-media-proxy:8080/v1/images/generations \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy cabin in the mountains at sunset",
    "model": "google/gemini-2.5-flash-image-preview",
    "size": "1792x1024",
    "quality": "hd",
    "n": 1
  }'
```

### Transcribe audio

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -F "file=@sample.wav" \
  -F "model=google/gemini-2.5-flash" \
  -F "response_format=json"
```

### Translate audio into English

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/translations \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -F "file=@sample.mp3" \
  -F "model=google/gemini-2.5-flash" \
  -F "response_format=verbose_json"
```

### Generate speech

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/speech \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from OpenRouter through an OpenAI-compatible proxy.",
    "model": "openai/gpt-4o-audio-preview",
    "voice": "alloy",
    "response_format": "wav"
  }' > speech.wav
```

## Limitations

- Models are passed through unchanged. The client must use OpenRouter model IDs
  that actually support the requested modality.
- Image `response_format: "url"` is still ignored. The proxy only returns
  `b64_json`.
- `n > 1` for images still means multiple parallel upstream requests and costs
  `n` times the credits.
- Audio transcription and translation are prompt-shaped onto chat completions.
  Structured responses such as `verbose_json` and `diarized_json` are best
  effort and depend on model behavior.
- Audio transcription and translation streaming is not implemented. The proxy
  always returns a final response body.
- `stream_format: "sse"` for speech is supported only as a simple SSE stream of
  `{ audio, transcript? }` chunks, not a guaranteed byte-for-byte OpenAI event
  schema.
- `instructions` and `speed` for speech are translated into prompt guidance,
  because OpenRouter exposes only `voice` and `format` as structured audio
  output controls in the documented flow.
- Masks on image edits remain best effort because OpenRouter models interpret
  them as additional image context rather than a native OpenAI mask primitive.

[oai-images]: https://platform.openai.com/docs/api-reference/images
