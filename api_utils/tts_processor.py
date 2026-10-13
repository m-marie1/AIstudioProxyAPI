"""
TTS Request Processor Module

Handles TTS request processing separately from chat requests.
"""

import asyncio
import logging
import os
from asyncio import Future
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import Response

from browser_utils.tts_controller import TTSController, TTSGenerationError
from logging_utils import set_request_id, set_source
from models import ClientDisconnectedError, TTSRequest

from .client_connection import (
    check_client_connection as _check_client_connection,
)
from .error_utils import (
    client_disconnected,
    server_error,
)


def _make_tts_error_detail(
    req_id: str,
    code: str,
    message: str,
    error_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
            "request_id": req_id,
        }
    }
    if extra:
        payload["error"]["details"] = extra
    return payload


def _make_tts_http_exception(
    req_id: str,
    status_code: int,
    code: str,
    message: str,
    error_type: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    retry_after: Optional[int] = None,
) -> HTTPException:
    headers: Optional[Dict[str, str]] = None
    if retry_after is not None:
        headers = {"Retry-After": str(retry_after)}
    return HTTPException(
        status_code=status_code,
        detail=_make_tts_error_detail(req_id, code, message, error_type, extra),
        headers=headers,
    )


def _build_tts_model_fallback_order(requested_model: str) -> List[str]:
    fallback_models = [
        "gemini-2.5-pro-preview-tts",
        "gemini-2.5-flash-preview-tts",
    ]
    ordered = [requested_model] if requested_model else []
    for model in fallback_models:
        if model not in ordered:
            ordered.append(model)
    return ordered


async def _switch_auth_profile_for_tts(req_id: str, logger: logging.Logger) -> str:
    """Switch auth profile by fully reconnecting browser/page with next saved profile."""
    from api_utils.auth_manager import auth_manager

    auth_manager.mark_profile_failed()
    next_profile = await auth_manager.get_next_profile()
    return await _activate_auth_profile_for_tts(next_profile, logger)


async def _activate_auth_profile_for_tts(profile_path: str, logger: logging.Logger) -> str:
    """Activate specified auth profile by reconnecting browser/page state."""
    from api_utils.server_state import state
    from browser_utils.initialization.core import (
        close_page_logic,
        enable_temporary_chat_mode,
        initialize_page_logic,
    )
    from browser_utils.model_management import (
        _handle_initial_model_state_and_storage,  # pyright: ignore[reportPrivateUsage]
    )
    from config import get_environment_variable

    os.environ["ACTIVE_AUTH_JSON_PATH"] = profile_path

    logger.warning(
        f"[TTS] Activating auth profile: {os.path.basename(profile_path)}"
    )

    await close_page_logic()

    if state.browser_instance and state.browser_instance.is_connected():
        await state.browser_instance.close()
        state.is_browser_connected = False

    ws_endpoint = get_environment_variable("CAMOUFOX_WS_ENDPOINT")
    if not ws_endpoint:
        raise RuntimeError("CAMOUFOX_WS_ENDPOINT not available for reconnection")

    if not state.playwright_manager:
        raise RuntimeError("Playwright manager not available")

    state.browser_instance = await state.playwright_manager.firefox.connect(
        ws_endpoint, timeout=30000
    )
    state.is_browser_connected = True

    state.page_instance, state.is_page_ready = await initialize_page_logic(
        state.browser_instance,
        storage_state_path=profile_path,
    )

    if not state.page_instance or not state.is_page_ready:
        raise RuntimeError("Page initialization failed after auth profile switch")

    await _handle_initial_model_state_and_storage(state.page_instance)
    await enable_temporary_chat_mode(state.page_instance)
    return profile_path


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
    logger.info(f"[TTS] Starting TTS request processing (Model: {request.model})")

    try:
        # Check client connection
        if not await _check_client_connection(req_id, http_request):
            logger.info("[TTS] Client disconnected")
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

        # Avoid stale stream-proxy fail-fast payloads from previous requests.
        from api_utils.utils_ext.stream import clear_stream_queue

        await clear_stream_queue()

        model_order = _build_tts_model_fallback_order(request.model)

        from api_utils.auth_manager import auth_manager

        # Per-request account rotation (round-robin)
        selected_profile = await auth_manager.get_next_profile_round_robin()
        current_active_profile = os.environ.get("ACTIVE_AUTH_JSON_PATH")
        needs_profile_switch = (
            not current_active_profile
            or os.path.basename(current_active_profile)
            != os.path.basename(selected_profile)
        )
        if needs_profile_switch:
            await _activate_auth_profile_for_tts(selected_profile, logger)
        else:
            logger.info(
                f"[TTS] Reusing already-active round-robin profile: {os.path.basename(selected_profile)}"
            )

        available_profiles = await auth_manager.get_available_profiles()
        max_auth_cycles = max(1, len(available_profiles) + 1)
        auth_cycle = 0
        tried_profiles: List[str] = []
        audio_data: Optional[bytes] = None

        while auth_cycle < max_auth_cycles:
            current_profile = auth_manager.current_profile
            tried_profiles.append(current_profile or "current-session")

            exhausted_models: List[str] = []

            # Re-read active page instance after profile switch
            page = state.page_instance
            if not page or page.is_closed():
                raise _make_tts_http_exception(
                    req_id,
                    503,
                    "tts_page_not_available",
                    "AI Studio page not available for TTS",
                    "service_unavailable_error",
                    retry_after=20,
                )

            tts_controller = TTSController(page, logger, req_id)

            for model_id in model_order:
                request_for_model = request.model_copy(update={"model": model_id})
                logger.info(
                    f"[TTS] Trying model: {model_id} (account cycle {auth_cycle + 1}/{max_auth_cycles})"
                )
                try:
                    audio_data = await tts_controller.process_tts_request(
                        request_for_model, disconnect_checker
                    )
                    if audio_data:
                        break
                except TTSGenerationError as tts_err:
                    if tts_err.code == "tts_too_many_requests":
                        exhausted_models.append(model_id)
                        logger.warning(
                            f"[TTS] Upstream returned 429 for model {model_id}, trying next model"
                        )
                        continue

                    status_by_code = {
                        "tts_upstream_bad_request": 400,
                        "tts_upstream_error": 502,
                        "tts_generation_timeout": 504,
                        "tts_navigation_failed": 502,
                        "tts_run_failed": 502,
                        "tts_audio_retrieval_failed": 502,
                        "tts_generation_failed": 502,
                    }
                    status = status_by_code.get(tts_err.code, 500)
                    error_type = "upstream_error"
                    if status == 400:
                        error_type = "invalid_request_error"
                    raise _make_tts_http_exception(
                        req_id,
                        status,
                        tts_err.code,
                        tts_err.message,
                        error_type,
                        extra={"model": model_id, **tts_err.details},
                    )

            if audio_data:
                break

            all_models_exhausted = len(exhausted_models) == len(model_order)

            if all_models_exhausted:
                if auth_cycle + 1 >= max_auth_cycles:
                    raise _make_tts_http_exception(
                        req_id,
                        429,
                        "tts_all_accounts_quota_exhausted",
                        "All configured accounts exhausted TTS daily quota for available models",
                        "rate_limit_error",
                        extra={
                            "models_tried": model_order,
                            "profiles_tried": tried_profiles,
                        },
                        retry_after=3600,
                    )

                try:
                    await _switch_auth_profile_for_tts(req_id, logger)
                    auth_cycle += 1
                    continue
                except RuntimeError as switch_err:
                    if "exhausted" in str(switch_err).lower():
                        raise _make_tts_http_exception(
                            req_id,
                            429,
                            "tts_auth_profiles_exhausted",
                            "No remaining authentication profiles available for TTS fallback",
                            "rate_limit_error",
                            extra={"profiles_tried": tried_profiles},
                            retry_after=3600,
                        )
                    raise _make_tts_http_exception(
                        req_id,
                        503,
                        "tts_auth_switch_failed",
                        f"Failed to switch auth profile for TTS: {switch_err}",
                        "service_unavailable_error",
                        retry_after=30,
                    )

            # Non-quota path with no audio data
            raise _make_tts_http_exception(
                req_id,
                502,
                "tts_no_audio_data",
                "TTS generation completed without audio output",
                "upstream_error",
                extra={"models_tried": model_order},
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

            logger.info(f"[TTS] TTS request processed successfully ({len(audio_data)} bytes)")
        else:
            raise _make_tts_http_exception(
                req_id,
                502,
                "tts_no_audio_data",
                "TTS generation returned no audio data",
                "upstream_error",
            )

    except ClientDisconnectedError as disco_err:
        logger.info(f"[TTS] Client disconnected: {disco_err}")
        if not result_future.done():
            result_future.set_exception(
                client_disconnected(req_id, "Client disconnected during TTS processing")
            )
    except asyncio.CancelledError:
        logger.info("[TTS] Request cancelled")
        if not result_future.done():
            result_future.cancel()
        raise
    except HTTPException as http_err:
        logger.warning(f"[TTS] HTTP exception: {http_err.detail}")
        if not result_future.done():
            result_future.set_exception(http_err)
    except Exception as e:
        logger.exception(f"[TTS] Processing failed: {e}")
        if not result_future.done():
            result_future.set_exception(
                server_error(req_id, f"TTS processing failed: {e}")
            )
