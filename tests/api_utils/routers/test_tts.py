"""Tests for TTS router endpoints."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from api_utils.routers.tts import audio_speech, list_tts_models, list_tts_voices
from models import TTSRequest


@pytest.mark.asyncio
async def test_list_tts_voices():
    """Test list_tts_voices endpoint returns voices correctly."""
    logger = MagicMock()

    response = await list_tts_voices(logger=logger)

    # Response should be a JSONResponse
    assert response.status_code == 200

    # Parse the body
    body = json.loads(response.body)

    assert "voices" in body
    assert len(body["voices"]) > 0

    # Check structure of a voice
    voice = body["voices"][0]
    assert "voice_id" in voice
    assert "name" in voice
    assert "description" in voice


@pytest.mark.asyncio
async def test_list_tts_models():
    """Test list_tts_models endpoint returns models correctly."""
    logger = MagicMock()

    response = await list_tts_models(logger=logger)

    # Response should be a JSONResponse
    assert response.status_code == 200

    # Parse the body
    body = json.loads(response.body)

    assert "object" in body
    assert body["object"] == "list"
    assert "data" in body
    assert len(body["data"]) >= 2

    # Check structure of a model
    model = body["data"][0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"
    assert "type" in model
    assert model["type"] == "tts"


@pytest.mark.asyncio
async def test_list_tts_voices_contains_zephyr():
    """Test that Zephyr voice is in the list."""
    logger = MagicMock()

    response = await list_tts_voices(logger=logger)

    body = json.loads(response.body)
    voice_names = [v["name"] for v in body["voices"]]

    assert "Zephyr" in voice_names


@pytest.mark.asyncio
async def test_list_tts_models_contains_expected_models():
    """Test that expected TTS models are in the list."""
    logger = MagicMock()

    response = await list_tts_models(logger=logger)

    body = json.loads(response.body)
    model_ids = [m["id"] for m in body["data"]]

    assert "gemini-2.5-pro-preview-tts" in model_ids
    assert "gemini-2.5-flash-preview-tts" in model_ids


@pytest.mark.asyncio
async def test_audio_speech_invalid_request_returns_structured_error():
    """Missing input/speakers should return explicit invalid_request_error payload."""
    request = TTSRequest(model="gemini-2.5-pro-preview-tts", input=None, speakers=None)

    response = await audio_speech(
        request=request,
        http_request=MagicMock(),
        logger=MagicMock(),
        request_queue=MagicMock(),
        server_state={},
        worker_task=MagicMock(),
    )

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["code"] == "tts_invalid_request"
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_audio_speech_preserves_structured_http_exception_from_worker():
    """Structured worker HTTPException details should pass through unchanged."""

    class ImmediateErrorQueue:
        async def put(self, item):
            item["result_future"].set_exception(
                HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": "All configured accounts exhausted TTS daily quota",
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "tts_all_accounts_quota_exhausted",
                            "request_id": "tts-test",
                        }
                    },
                )
            )

    server_state = {
        "is_initializing": False,
        "is_playwright_ready": True,
        "is_page_ready": True,
        "is_browser_connected": True,
    }

    worker_task = MagicMock()
    worker_task.done.return_value = False

    response = await audio_speech(
        request=TTSRequest(
            model="gemini-2.5-pro-preview-tts",
            input="hello world",
            voice="Zephyr",
        ),
        http_request=MagicMock(),
        logger=MagicMock(),
        request_queue=ImmediateErrorQueue(),
        server_state=server_state,
        worker_task=worker_task,
    )

    assert response.status_code == 429
    body = json.loads(response.body)
    assert body["error"]["code"] == "tts_all_accounts_quota_exhausted"
    assert body["error"]["type"] == "rate_limit_error"
