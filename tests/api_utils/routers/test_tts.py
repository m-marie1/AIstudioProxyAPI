"""Tests for TTS router endpoints."""

from unittest.mock import MagicMock

import pytest

from api_utils.routers.tts import list_tts_models, list_tts_voices


@pytest.mark.asyncio
async def test_list_tts_voices():
    """Test list_tts_voices endpoint returns voices correctly."""
    logger = MagicMock()

    response = await list_tts_voices(logger=logger)

    # Response should be a JSONResponse
    assert response.status_code == 200

    # Parse the body
    import json

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
    import json

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

    import json

    body = json.loads(response.body)
    voice_names = [v["name"] for v in body["voices"]]

    assert "Zephyr" in voice_names


@pytest.mark.asyncio
async def test_list_tts_models_contains_expected_models():
    """Test that expected TTS models are in the list."""
    logger = MagicMock()

    response = await list_tts_models(logger=logger)

    import json

    body = json.loads(response.body)
    model_ids = [m["id"] for m in body["data"]]

    assert "gemini-2.5-pro-preview-tts" in model_ids
    assert "gemini-2.5-flash-preview-tts" in model_ids
