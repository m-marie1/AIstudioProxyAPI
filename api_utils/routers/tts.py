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
        f"收到 /v1/audio/speech 请求 (Model={request.model}, Voice={request.voice})"
    )

    # Validate request
    if not request.input and not request.speakers:
        raise HTTPException(
            status_code=400,
            detail="Either 'input' (for single-speaker) or 'speakers' (for multi-speaker) must be provided.",
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
        raise HTTPException(status_code=504, detail=f"[{req_id}] TTS请求处理超时。")
    except asyncio.CancelledError:
        logger.info(f"TTS请求被客户端取消: {req_id}")
        raise
    except HTTPException as http_exc:
        if http_exc.status_code == 499:
            logger.info(f"客户端断开连接: {http_exc.detail}")
        else:
            logger.warning(f"HTTP异常: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception("等待TTS响应时出错")
        raise HTTPException(status_code=500, detail=f"[{req_id}] 服务器内部错误: {e}")


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
