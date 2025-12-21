"""
TTS Request Processor Module

Handles TTS request processing separately from chat requests.
"""

import asyncio
import logging
from asyncio import Future

from fastapi import HTTPException, Request
from fastapi.responses import Response

from browser_utils.tts_controller import TTSController
from logging_utils import set_request_id, set_source
from models import ClientDisconnectedError, TTSRequest

from .client_connection import (
    check_client_connection as _check_client_connection,
)
from .error_utils import (
    client_disconnected,
    server_error,
)


class TTSClientDisconnectChecker:
    """
    Helper class to check client disconnection state.

    Stores the http_request and provides a synchronous check method
    that can be used in callbacks where async is not available.
    """

    def __init__(self, req_id: str, http_request: Request, logger: logging.Logger):
        self.req_id = req_id
        self.http_request = http_request
        self.logger = logger
        self._disconnected = False

    async def check_async(self) -> bool:
        """Async check for client disconnection."""
        try:
            self._disconnected = await self.http_request.is_disconnected()
            return self._disconnected
        except Exception:
            return False

    def __call__(self, stage: str) -> bool:
        """
        Synchronous check for client disconnection.

        Returns the last known disconnection state.
        The async check should be called periodically to update the state.
        """
        if self._disconnected:
            self.logger.debug(f"[TTS] Client disconnected at stage: {stage}")
        return self._disconnected


async def process_tts_request(
    req_id: str,
    request: TTSRequest,
    http_request: Request,
    result_future: Future[Response],
) -> None:
    """
    Process a TTS request and return audio data.

    Args:
        req_id: Request ID
        request: TTS request object
        http_request: FastAPI HTTP request
        result_future: Future to set with the result
    """
    set_request_id(req_id)
    set_source("TTS-PROCESSOR")

    logger = logging.getLogger("AIStudioProxyServer")
    logger.info(f"[TTS] 开始处理TTS请求 (Model: {request.model})")

    try:
        # Check client connection
        if not await _check_client_connection(req_id, http_request):
            logger.info("[TTS] 客户端已断开连接")
            if not result_future.done():
                result_future.set_exception(
                    client_disconnected(req_id, "Client disconnected before TTS processing")
                )
            return

        # Get page instance
        from api_utils.server_state import state

        page = state.page_instance
        if not page or page.is_closed():
            raise HTTPException(
                status_code=503,
                detail=f"[{req_id}] AI Studio page not available for TTS",
            )

        # Create disconnect checker with proper async support
        disconnect_checker = TTSClientDisconnectChecker(req_id, http_request, logger)

        # Create TTS controller
        tts_controller = TTSController(page, logger, req_id)

        # Process the TTS request
        audio_data = await tts_controller.process_tts_request(
            request, disconnect_checker
        )

        if audio_data:
            # Determine content type based on response format
            content_type_map = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav",
                "pcm": "audio/pcm",
            }
            content_type = content_type_map.get(request.response_format, "audio/mpeg")

            # Create response with audio data
            response = Response(
                content=audio_data,
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',
                },
            )

            if not result_future.done():
                result_future.set_result(response)

            logger.info(f"[TTS] TTS请求处理成功 ({len(audio_data)} bytes)")
        else:
            raise Exception("TTS generation returned no audio data")

    except ClientDisconnectedError as disco_err:
        logger.info(f"[TTS] 客户端已断开: {disco_err}")
        if not result_future.done():
            result_future.set_exception(
                client_disconnected(req_id, "Client disconnected during TTS processing")
            )
    except asyncio.CancelledError:
        logger.info("[TTS] 请求已取消")
        if not result_future.done():
            result_future.cancel()
        raise
    except HTTPException as http_err:
        logger.warning(f"[TTS] HTTP异常: {http_err.detail}")
        if not result_future.done():
            result_future.set_exception(http_err)
    except Exception as e:
        logger.exception(f"[TTS] 处理失败: {e}")
        if not result_future.done():
            result_future.set_exception(
                server_error(req_id, f"TTS processing failed: {e}")
            )
