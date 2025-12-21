"""Tests for TTS models."""

from models.tts import (
    AVAILABLE_VOICES,
    TTS_MODELS,
    TTSRequest,
    TTSResponse,
    TTSSpeaker,
    get_default_tts_model,
    is_tts_model,
)


def test_tts_speaker_model():
    """Test TTSSpeaker model validation."""
    speaker = TTSSpeaker(text="Hello world")
    assert speaker.text == "Hello world"
    assert speaker.voice == "Zephyr"  # Default voice
    assert speaker.name is None

    speaker_custom = TTSSpeaker(text="Custom text", voice="Kore", name="Speaker 1")
    assert speaker_custom.voice == "Kore"
    assert speaker_custom.name == "Speaker 1"


def test_tts_request_single_speaker():
    """Test TTSRequest for single-speaker mode."""
    request = TTSRequest(input="Hello world", voice="Zephyr")
    assert request.input == "Hello world"
    assert request.voice == "Zephyr"
    assert request.model == "gemini-2.5-pro-preview-tts"  # Default model
    assert request.speakers is None
    assert request.response_format == "mp3"  # Default format
    assert request.speed == 1.0  # Default speed


def test_tts_request_multi_speaker():
    """Test TTSRequest for multi-speaker mode."""
    speakers = [
        TTSSpeaker(text="Hello!", voice="Zephyr"),
        TTSSpeaker(text="Hi there!", voice="Kore"),
    ]
    request = TTSRequest(speakers=speakers)
    assert request.speakers is not None
    assert len(request.speakers) == 2
    assert request.speakers[0].voice == "Zephyr"
    assert request.speakers[1].voice == "Kore"


def test_tts_request_with_style():
    """Test TTSRequest with style instructions."""
    request = TTSRequest(
        input="Welcome to the show",
        voice="Puck",
        style_instructions="Read aloud in a warm and friendly tone",
    )
    assert request.style_instructions == "Read aloud in a warm and friendly tone"


def test_tts_request_different_formats():
    """Test TTSRequest with different audio formats."""
    for fmt in ["mp3", "opus", "aac", "flac", "wav", "pcm"]:
        request = TTSRequest(input="Test", response_format=fmt)
        assert request.response_format == fmt


def test_tts_request_speed_limits():
    """Test TTSRequest speed parameter limits."""
    request = TTSRequest(input="Test", speed=0.25)
    assert request.speed == 0.25

    request = TTSRequest(input="Test", speed=4.0)
    assert request.speed == 4.0


def test_tts_response_model():
    """Test TTSResponse model."""
    response = TTSResponse()
    assert response.error is None

    response_error = TTSResponse(error={"code": "invalid_request", "message": "Bad"})
    assert response_error.error is not None
    assert response_error.error["code"] == "invalid_request"


def test_is_tts_model():
    """Test is_tts_model function."""
    assert is_tts_model("gemini-2.5-pro-preview-tts") is True
    assert is_tts_model("gemini-2.5-flash-preview-tts") is True
    assert is_tts_model("GEMINI-2.5-PRO-PREVIEW-TTS") is True  # Case insensitive
    assert is_tts_model("gemini-2.5-pro-preview") is False
    assert is_tts_model("gpt-4") is False
    assert is_tts_model("") is False
    assert is_tts_model(None) is False  # type: ignore


def test_get_default_tts_model():
    """Test get_default_tts_model function."""
    default = get_default_tts_model()
    assert default == "gemini-2.5-pro-preview-tts"
    assert default in TTS_MODELS


def test_tts_models_list():
    """Test TTS_MODELS list."""
    assert len(TTS_MODELS) >= 2
    assert "gemini-2.5-pro-preview-tts" in TTS_MODELS
    assert "gemini-2.5-flash-preview-tts" in TTS_MODELS


def test_available_voices():
    """Test AVAILABLE_VOICES list."""
    assert len(AVAILABLE_VOICES) > 0
    assert "Zephyr" in AVAILABLE_VOICES
    assert "Puck" in AVAILABLE_VOICES
    assert "Kore" in AVAILABLE_VOICES
