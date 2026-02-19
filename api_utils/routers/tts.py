"""
TTS (Text-to-Speech) API Router

Provides OpenAI-compatible /v1/audio/speech endpoint for TTS generation
using Gemini TTS models via AI Studio.
"""

import asyncio
import logging
import random
import time
from asyncio import Future, Queue
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from config import get_environment_variable
from logging_utils import set_request_id, set_source
from models import TTSRequest

from ..dependencies import (
    get_logger,
    get_request_queue,
    get_server_state,
    get_worker_task,
)
from ..error_utils import service_unavailable


async def audio_speech(
    request: TTSRequest,
    http_request: Request,
    logger: logging.Logger = Depends(get_logger),
    request_queue: "Queue[Any]" = Depends(get_request_queue),
    server_state: dict = Depends(get_server_state),  # pyright: ignore[reportMissingTypeArgument]
    worker_task: Any = Depends(get_worker_task),
) -> Response:
    """
    Generate audio from text using Gemini TTS.

    OpenAI-compatible endpoint: POST /v1/audio/speech

    Request body:
    - model: TTS model ID (gemini-2.5-pro-preview-tts or gemini-2.5-flash-preview-tts)
    - input: Text to synthesize (single-speaker mode)
    - voice: Voice name (e.g., "Zephyr", "Puck", "Kore")
    - speakers: Array of speaker objects for multi-speaker mode (optional)
    - style_instructions: Style instructions for the TTS (optional)
    - response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
    - speed: Speed of the generated audio (0.25-4.0)

    Returns: Audio file as binary response
    """
    req_id = "tts-" + "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=7)
    )

    # Set logging context
    set_request_id(req_id)
    set_source("TTS-API")

    logger.info(
        f"Received /v1/audio/speech request (Model={request.model}, Voice={request.voice})"
    )

    def _router_tts_error(
        status_code: int,
        code: str,
        message: str,
        error_type: str = "server_error",
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": code,
                    "request_id": req_id,
                }
            },
        )

    # Validate request
    if not request.input and not request.speakers:
        return _router_tts_error(
            400,
            "tts_invalid_request",
            "Either 'input' (for single-speaker) or 'speakers' (for multi-speaker) must be provided.",
            "invalid_request_error",
        )

    launch_mode = get_environment_variable("LAUNCH_MODE", "unknown")
    browser_page_critical = launch_mode != "direct_debug_no_browser"

    is_service_unavailable = (
        server_state["is_initializing"]
        or not server_state["is_playwright_ready"]
        or (
            browser_page_critical
            and (
                not server_state["is_page_ready"]
                or not server_state["is_browser_connected"]
            )
        )
        or not worker_task
        or worker_task.done()
    )

    if is_service_unavailable:
        raise service_unavailable(req_id)

    # Create a future for the result
    result_future: Future[Response] = Future()

    # Queue the TTS request
    queue_item = {
        "req_id": req_id,
        "request_type": "tts",  # Mark as TTS request
        "request_data": request,
        "http_request": http_request,
        "result_future": result_future,
        "enqueue_time": time.time(),
        "cancelled": False,
    }
    await request_queue.put(queue_item)

    try:
        # TTS requests typically take longer than chat, use extended timeout
        timeout_seconds = 300  # 5 minutes for TTS
        return await asyncio.wait_for(result_future, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return _router_tts_error(
            504,
            "tts_processing_timeout",
            "TTS request processing timed out",
            "timeout_error",
        )
    except asyncio.CancelledError:
        logger.info(f"TTS request cancelled by client: {req_id}")
        raise
    except HTTPException as http_exc:
        detail = http_exc.detail
        if http_exc.status_code == 499:
            logger.info(f"Client disconnected: {detail}")
        elif isinstance(detail, dict) and "error" in detail:
            return JSONResponse(
                status_code=http_exc.status_code,
                content=detail,
                headers=http_exc.headers,
            )
        else:
            logger.warning(f"HTTP exception: {detail}")
            return _router_tts_error(
                http_exc.status_code,
                "tts_http_error",
                str(detail),
                "upstream_error",
            )
        raise
    except Exception as e:
        logger.exception("Error while waiting for TTS response")
        return _router_tts_error(
            500,
            "tts_internal_error",
            f"Internal TTS server error: {e}",
            "server_error",
        )


async def list_tts_voices(
    logger: logging.Logger = Depends(get_logger),
) -> JSONResponse:
    """
    List available TTS voices.

    Returns a list of available voice options for TTS.
    """
    from models.tts import AVAILABLE_VOICES

    voices = [
        {
            "voice_id": voice.lower(),
            "name": voice,
            "preview_url": None,  # No preview URLs available
            "description": f"Gemini TTS voice: {voice}",
        }
        for voice in AVAILABLE_VOICES
    ]

    return JSONResponse(
        content={
            "voices": voices,
        }
    )


async def list_tts_models(
    logger: logging.Logger = Depends(get_logger),
) -> JSONResponse:
    """
    List available TTS models.

    Returns a list of TTS model options.
    """
    from models.tts import TTS_MODELS

    models = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "type": "tts",
        }
        for model_id in TTS_MODELS
    ]

    return JSONResponse(
        content={
            "object": "list",
            "data": models,
        }
    )
