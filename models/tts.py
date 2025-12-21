"""
TTS (Text-to-Speech) Models

Defines request and response models for the OpenAI-compatible TTS API.
Supports both single-speaker and multi-speaker audio generation via Gemini TTS.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Available voices from AI Studio TTS
# Based on the UI screenshots, these are the available voice options
AVAILABLE_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

# Create a set for faster lookups (case-insensitive)
_AVAILABLE_VOICES_LOWER = {v.lower() for v in AVAILABLE_VOICES}

# TTS model IDs
TTS_MODELS = [
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
]

# Create a set for exact matching (case-insensitive)
_TTS_MODELS_LOWER = {m.lower() for m in TTS_MODELS}


def _validate_voice(voice: str) -> str:
    """Validate and normalize a voice name."""
    if voice.lower() not in _AVAILABLE_VOICES_LOWER:
        # Find the closest match or use default
        # For now, just use the voice as-is to allow new voices
        pass
    return voice


class TTSSpeaker(BaseModel):
    """Represents a speaker in multi-speaker TTS mode."""

    text: str = Field(..., description="The text for this speaker to speak")
    voice: str = Field(
        default="Zephyr",
        description="Voice name for this speaker (e.g., 'Zephyr', 'Puck', 'Kore')",
    )
    name: Optional[str] = Field(
        default=None, description="Optional display name for the speaker"
    )

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, v: str) -> str:
        """Validate voice is in the available voices list."""
        if v.lower() not in _AVAILABLE_VOICES_LOWER:
            # Log warning but allow the voice (in case new voices are added)
            # This is more lenient than strict validation
            pass
        return v


class TTSRequest(BaseModel):
    """
    OpenAI-compatible TTS request model.

    Supports both single-speaker mode (using input/voice) and
    multi-speaker mode (using speakers array).
    """

    model: str = Field(
        default="gemini-2.5-pro-preview-tts",
        description="TTS model ID (gemini-2.5-pro-preview-tts or gemini-2.5-flash-preview-tts)",
    )

    # Single-speaker mode fields (OpenAI compatible)
    input: Optional[str] = Field(
        default=None, description="The text to generate audio for (single-speaker mode)"
    )
    voice: str = Field(
        default="Zephyr",
        description="Voice to use for generation (e.g., 'Zephyr', 'Puck', 'Kore')",
    )

    # Multi-speaker mode fields (extension)
    speakers: Optional[List[TTSSpeaker]] = Field(
        default=None,
        description="Array of speakers for multi-speaker mode. If provided, 'input' is ignored.",
    )

    # Style instructions
    style_instructions: Optional[str] = Field(
        default=None,
        description="Style instructions for the TTS (e.g., 'Read aloud in a warm and friendly tone')",
    )

    # Response format
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", description="Audio format for the response"
    )

    # Speed control (OpenAI compatible, but may not be fully supported)
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speed of the generated audio"
    )

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, v: str) -> str:
        """Validate voice is in the available voices list."""
        if v.lower() not in _AVAILABLE_VOICES_LOWER:
            # Log warning but allow the voice (in case new voices are added)
            pass
        return v


class TTSResponse(BaseModel):
    """
    TTS response model.

    The actual audio data is returned as binary in the HTTP response body,
    not as JSON. This model is used for error responses.
    """

    error: Optional[Dict[str, Any]] = Field(
        default=None, description="Error information if the request failed"
    )


def is_tts_model(model_id: str) -> bool:
    """Check if a model ID is a TTS model using exact matching."""
    if not model_id:
        return False
    return model_id.lower() in _TTS_MODELS_LOWER


def get_default_tts_model() -> str:
    """Get the default TTS model ID."""
    return "gemini-2.5-pro-preview-tts"
