"""
openrouter_image_proxy.py

A minimal FastAPI proxy that translates OpenAI-compatible image-generation
requests (/images/generations, /images/edits) into OpenRouter's
chat/completions API format.

Designed to run as an internal Docker-network sidecar in front of Open WebUI.
"""

import os
import json
import time
import uuid
import re
import base64
import logging
import asyncio

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---- Configuration ---------------------------------------------------------

app = FastAPI()

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT", "120"))
# Comma-separated.  Use "image" alone for image-only models (Flux, Sourceful).
DEFAULT_MODALITIES = os.getenv("DEFAULT_MODALITIES", "image,text")

# ---- Logging ---------------------------------------------------------------

_raw_level = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, _raw_level, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("image_proxy")

# ---- Constants / mappings --------------------------------------------------

# OpenAI pixel dimensions -> OpenRouter aspect ratios
SIZE_TO_ASPECT: dict[str, str] = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "256x256":   "1:1",
    "512x512":   "1:1",
}

# OpenAI quality labels -> OpenRouter image_size tokens
QUALITY_TO_IMAGE_SIZE: dict[str, str] = {
    "low":      "1K",
    "standard": "1K",
    "medium":   "2K",
    "hd":       "2K",
    "high":     "4K",
}

DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,(.*)", re.DOTALL)

# ---- Helpers ---------------------------------------------------------------


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


def _modalities() -> list[str]:
    return [m.strip() for m in DEFAULT_MODALITIES.split(",") if m.strip()]


def build_image_config(size: str | None, quality: str | None) -> dict:
    cfg: dict = {}
    if size and size != "auto":
        ar = SIZE_TO_ASPECT.get(size)
        if ar:
            cfg["aspect_ratio"] = ar
    if quality and quality != "auto":
        isz = QUALITY_TO_IMAGE_SIZE.get(quality)
        if isz:
            cfg["image_size"] = isz
    return cfg


def upstream_headers(request: Request) -> dict:
    """Forward only the Authorization header to OpenRouter."""
    hdrs: dict[str, str] = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        hdrs["Authorization"] = auth
    return hdrs


def extract_images(data: dict) -> list[dict]:
    """
    Pull base64 images out of an OpenRouter chat/completions response
    and reshape them into OpenAI ImagesResponse.data entries.
    """
    images: list[dict] = []
    for choice in data.get("choices", []):
        msg = choice.get("message") or choice.get("delta") or {}
        revised: str | None = None
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            revised = content.strip()
        for img in msg.get("images", []):
            url = img.get("image_url", {}).get("url", "")
            match = DATA_URL_RE.match(url)
            if match:
                entry: dict = {"b64_json": match.group(1)}
                if revised:
                    entry["revised_prompt"] = revised
                images.append(entry)
    return images


def _augment_prompt(
    prompt: str,
    style: str | None = None,
    background: str | None = None,
) -> str:
    """Append OpenAI-specific style / background hints to the prompt text."""
    parts = [prompt]
    if style == "natural":
        parts.append("Use a natural, realistic style.")
    elif style == "vivid":
        parts.append("Use a vivid, dramatic style.")
    if background == "transparent":
        parts.append("The image should have a transparent background.")
    return " ".join(parts)


async def _call_upstream(
    client: httpx.AsyncClient,
    body: dict,
    headers: dict,
    rid: str,
    idx: int,
) -> tuple[dict | None, tuple[int, dict] | None]:
    """
    POST to OpenRouter chat/completions.
    Returns (success_json, None) or (None, (status_code, error_body)).
    """
    url = f"{OPENROUTER_URL}/chat/completions"
    try:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code == 200:
            return resp.json(), None
        logger.warning(
            "rid=%s idx=%s upstream_status=%s body=%s",
            rid, idx, resp.status_code, resp.text[:500],
        )
        try:
            err_body = resp.json()
        except Exception:
            err_body = {
                "error": {"message": resp.text[:500], "type": "upstream_error"}
            }
        return None, (resp.status_code, err_body)
    except httpx.TimeoutException:
        logger.error("rid=%s idx=%s event=timeout", rid, idx)
        return None, (
            504,
            {"error": {"message": "Upstream request timed out", "type": "timeout_error"}},
        )
    except Exception as e:
        logger.exception("rid=%s idx=%s event=exception", rid, idx)
        return None, (
            502,
            {"error": {"message": f"Upstream request failed: {e}", "type": "proxy_error"}},
        )


# ---- Routes ----------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- /images/generations ---------------------------------------------------


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def generations(request: Request):
    rid = _request_id(request)
    body = await request.json()

    prompt = body.get("prompt", "")
    model = body.get("model", "")
    n = max(1, min(body.get("n") or 1, 10))
    size = body.get("size")
    quality = body.get("quality")
    style = body.get("style")
    background = body.get("background")

    full_prompt = _augment_prompt(prompt, style, background)
    image_config = build_image_config(size, quality)

    or_body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "modalities": _modalities(),
    }
    if image_config:
        or_body["image_config"] = image_config

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=generations model=%s n=%s size=%s quality=%s",
        rid, model, n, size, quality,
    )

    all_images: list[dict] = []
    last_error: tuple[int, dict] | None = None

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        tasks = [_call_upstream(client, or_body, headers, rid, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        for success, error in results:
            if success:
                all_images.extend(extract_images(success))
            elif error:
                last_error = error

    if not all_images:
        status, err_body = last_error or (
            502,
            {"error": {"message": "No images returned by upstream", "type": "upstream_error"}},
        )
        logger.error("rid=%s endpoint=generations error=no_images status=%s", rid, status)
        return JSONResponse(status_code=status, content=err_body)

    logger.info("rid=%s endpoint=generations images_returned=%s", rid, len(all_images))
    return JSONResponse(content={"created": int(time.time()), "data": all_images})


# ---- /images/edits ---------------------------------------------------------


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def edits(request: Request):
    rid = _request_id(request)
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        form = await request.form()
        prompt = str(form.get("prompt") or "")
        model = str(form.get("model") or "")
        n = max(1, min(int(form.get("n") or 1), 10))
        size = str(form.get("size") or "") or None
        quality = str(form.get("quality") or "") or None
        background = str(form.get("background") or "") or None

        EXPECTED_FILE_FIELDS = {"image", "image[]", "images", "images[]", "mask", "mask[]"}

        image_urls: list[str] = []
        for key, value in form.multi_items():
            if hasattr(value, "read") and key in EXPECTED_FILE_FIELDS:
                raw = await value.read()
                if raw:
                    ct = getattr(value, "content_type", None) or "image/png"
                    b64 = base64.b64encode(raw).decode()
                    image_urls.append(f"data:{ct};base64,{b64}")
    else:
        body = await request.json()
        prompt = body.get("prompt", "")
        model = body.get("model", "")
        n = max(1, min(body.get("n") or 1, 10))
        size = body.get("size")
        quality = body.get("quality")
        background = body.get("background")

        image_urls = []
        for img in body.get("images", []):
            url = img.get("image_url")
            if url:
                image_urls.append(url)
        mask = body.get("mask")
        if mask:
            mask_url = mask.get("image_url")
            if mask_url:
                image_urls.append(mask_url)

    # Build multimodal content parts for OpenRouter
    # OpenRouter recommends text first, then images.
    prompt_text = _augment_prompt(prompt, background=background)
    content_parts: list[dict] = [{"type": "text", "text": prompt_text}]
    content_parts.extend(
        {"type": "image_url", "image_url": {"url": u}} for u in image_urls
    )

    image_config = build_image_config(size, quality)

    or_body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "modalities": _modalities(),
    }
    if image_config:
        or_body["image_config"] = image_config

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=edits model=%s n=%s input_images=%s",
        rid, model, n, len(image_urls),
    )

    all_images: list[dict] = []
    last_error: tuple[int, dict] | None = None

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        tasks = [_call_upstream(client, or_body, headers, rid, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        for success, error in results:
            if success:
                all_images.extend(extract_images(success))
            elif error:
                last_error = error

    if not all_images:
        status, err_body = last_error or (
            502,
            {"error": {"message": "No images returned by upstream", "type": "upstream_error"}},
        )
        logger.error("rid=%s endpoint=edits error=no_images status=%s", rid, status)
        return JSONResponse(status_code=status, content=err_body)

    logger.info("rid=%s endpoint=edits images_returned=%s", rid, len(all_images))
    return JSONResponse(content={"created": int(time.time()), "data": all_images})
