"""
TTS Page Controller Module

Handles browser interactions for TTS (Text-to-Speech) functionality in AI Studio.
Manages the TTS-specific UI elements for both single-speaker and multi-speaker modes.
"""

import asyncio
import base64
import json
import logging
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import aiohttp
from playwright.async_api import Page as AsyncPage
from playwright.async_api import expect as expect_async

from logging_utils import set_request_id
from models import TTSRequest, TTSSpeaker

# TTS-specific selectors (based on AI Studio TTS UI)
TTS_MODEL_SELECTOR = "gemini-2.5-pro-preview-tts"

# Style instructions input (single-speaker mode)
TTS_STYLE_INSTRUCTIONS_SELECTOR = 'textarea[aria-label*="Style"], textarea[placeholder*="style" i], .style-instructions textarea'

# Single-speaker mode selectors - Text input field
TTS_TEXT_INPUT_SELECTOR = 'textarea[aria-label*="Text"], textarea[placeholder*="text" i], .text-input textarea'

# Voice dropdown selector - the mat-select element that opens voice list
# The dropdown panel has id like 'mat-select-0-panel', trigger is mat-select element
# In single-speaker: under "Voice" label on right panel
# In multi-speaker: under "Speaker 1 settings" and "Speaker 2 settings"
TTS_VOICE_DROPDOWN_SELECTOR = (
    'mat-select, '
    'div[role="combobox"], '
    '[aria-haspopup="listbox"]'
)

# Run button selector
TTS_RUN_BUTTON_SELECTOR = 'button[aria-label*="Run"], button:has-text("Run"), .run-button'

# Audio player/download selectors
TTS_AUDIO_PLAYER_SELECTOR = "audio, .audio-player audio"
TTS_DOWNLOAD_BUTTON_SELECTOR = 'button[aria-label*="Download"], button:has-text("Download"), a[download]'

# Multi-speaker mode selectors
# Raw structure textarea on the left panel - this is where we input formatted text
TTS_RAW_STRUCTURE_SELECTOR = (
    'textarea[aria-label*="Raw" i], '
    'textarea[aria-label*="structure" i], '
    '.raw-structure textarea, '
    'ms-tts-raw-input textarea, '
    'div:has(> span:has-text("Raw structure")) + * textarea, '
    'div:has(> div:has-text("Raw structure")) textarea'
)

# Add dialog button (kept for fallback)
TTS_ADD_DIALOG_BUTTON_SELECTOR = 'button:has-text("Add dialog"), button[aria-label*="Add dialog"]'
TTS_SPEAKER_INPUT_SELECTOR = ".speaker-dialog textarea, .dialog-input textarea"
TTS_SPEAKER_BADGE_SELECTOR = ".speaker-badge, .speaker-label"

# Speaker settings sections on right panel (for voice selection)
TTS_SPEAKER_1_SETTINGS_SELECTOR = 'div:has-text("Speaker 1 settings"), [aria-label*="Speaker 1"]'
TTS_SPEAKER_2_SETTINGS_SELECTOR = 'div:has-text("Speaker 2 settings"), [aria-label*="Speaker 2"]'

# Mode toggle selectors
TTS_SINGLE_SPEAKER_MODE_SELECTOR = 'button:has-text("Single-speaker"), [aria-label*="Single-speaker"]'
TTS_MULTI_SPEAKER_MODE_SELECTOR = 'button:has-text("Multi-speaker"), [aria-label*="Multi-speaker"]'

# Model selector in TTS page
TTS_MODEL_DROPDOWN_SELECTOR = ".model-selector, mat-select[aria-label*='Model']"


class TTSGenerationError(Exception):
    """Structured TTS generation error for upstream handling."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class TTSController:
    """Controller for TTS page interactions in AI Studio."""

    def __init__(self, page: AsyncPage, logger: logging.Logger, req_id: str):
        self.page = page
        self.logger = logger
        self.req_id = req_id

    async def _check_disconnect(
        self, check_client_disconnected: Callable[[str], bool], stage: str
    ) -> None:
        """Check if client has disconnected."""
        from models import ClientDisconnectedError

        if check_client_disconnected(stage):
            raise ClientDisconnectedError(f"Client disconnected at stage: {stage}")

    async def navigate_to_tts_page(self, model_id: str = TTS_MODEL_SELECTOR) -> bool:
        """
        Navigate to the TTS page in AI Studio.

        Args:
            model_id: The TTS model ID to use

        Returns:
            True if navigation was successful
        """
        set_request_id(self.req_id)
        self.logger.info(f"[TTS] Navigating to TTS page (Model: {model_id})")

        try:
            # The TTS page URL pattern in AI Studio
            # Based on how AI Studio structures URLs
            tts_url = f"https://aistudio.google.com/generate-speech?model={model_id}"
            #https://aistudio.google.com/prompts/new_tts?model=gemini-2.5-pro-preview-tts

            # Navigate to TTS page
            await self.page.goto(tts_url, wait_until="domcontentloaded", timeout=30000)

            # Wait for the page to be ready
            await asyncio.sleep(2)

            # Verify we're on the TTS page by checking for TTS-specific elements
            try:
                # Look for style instructions or text input
                style_input = self.page.locator(TTS_STYLE_INSTRUCTIONS_SELECTOR)
                text_input = self.page.locator(TTS_TEXT_INPUT_SELECTOR)

                has_style = await style_input.count() > 0
                has_text = await text_input.count() > 0

                if has_style or has_text:
                    self.logger.info("[TTS] TTS page loaded successfully")
                    return True
                else:
                    self.logger.warning("[TTS] TTS page elements not found, trying fallback checks")
                    # Check for run button as fallback
                    run_button = self.page.locator(TTS_RUN_BUTTON_SELECTOR)
                    if await run_button.count() > 0:
                        return True

            except Exception as e:
                self.logger.warning(f"[TTS] Page validation failed: {e}")

            return False

        except Exception as e:
            self.logger.error(f"[TTS] Failed to navigate to TTS page: {e}")
            return False

    async def set_tts_mode(self, is_multi_speaker: bool) -> bool:
        """
        Set TTS mode (single-speaker or multi-speaker).

        Args:
            is_multi_speaker: True for multi-speaker mode, False for single-speaker

        Returns:
            True if mode was set successfully
        """
        set_request_id(self.req_id)
        mode_name = "Multi-speaker" if is_multi_speaker else "Single-speaker"
        self.logger.info(f"[TTS] Setting TTS mode: {mode_name}")

        try:
            selector = (
                TTS_MULTI_SPEAKER_MODE_SELECTOR
                if is_multi_speaker
                else TTS_SINGLE_SPEAKER_MODE_SELECTOR
            )
            mode_button = self.page.locator(selector)

            if await mode_button.count() > 0:
                await mode_button.first.click(timeout=5000)
                await asyncio.sleep(0.5)
                self.logger.info(f"[TTS] {mode_name} mode enabled")
                return True
            else:
                self.logger.warning(f"[TTS] {mode_name} mode button not found")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] Failed to set mode: {e}")
            return False

    async def fill_style_instructions(self, style: Optional[str]) -> bool:
        """
        Fill in the style instructions field.

        Args:
            style: Style instructions text

        Returns:
            True if successful
        """
        if not style:
            return True

        set_request_id(self.req_id)
        self.logger.debug(f"[TTS] Filling style instructions ({len(style)} chars)")

        try:
            style_input = self.page.locator(TTS_STYLE_INSTRUCTIONS_SELECTOR).first

            if await style_input.count() > 0:
                await style_input.fill(style, timeout=5000)
                self.logger.debug("[TTS] Style instructions filled")
                return True
            else:
                self.logger.warning("[TTS] Style instructions input not found")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] Failed to fill style instructions: {e}")
            return False

    async def fill_text_input(self, text: str) -> bool:
        """
        Fill in the text input for single-speaker mode.

        Args:
            text: Text to synthesize

        Returns:
            True if successful
        """
        set_request_id(self.req_id)
        self.logger.debug(f"[TTS] Filling text input ({len(text)} chars)")

        try:
            text_input = self.page.locator(TTS_TEXT_INPUT_SELECTOR).first

            if await text_input.count() > 0:
                await text_input.fill(text, timeout=5000)
                self.logger.debug("[TTS] Text input filled")
                return True
            else:
                self.logger.warning("[TTS] Text input not found")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] Failed to fill text input: {e}")
            return False

    async def select_voice(self, voice: str, voice_index: int = 0) -> bool:
        """
        Select a voice from the dropdown.

        Args:
            voice: Voice name (e.g., "Zephyr")
            voice_index: Which voice dropdown to use (0 for single-speaker or Speaker 1, 1 for Speaker 2)

        Returns:
            True if successful
        """
        set_request_id(self.req_id)
        self.logger.info(f"[TTS] Selecting voice: {voice} (index: {voice_index})")

        try:
            # Find voice dropdowns (mat-select elements)
            voice_dropdowns = self.page.locator(TTS_VOICE_DROPDOWN_SELECTOR)
            dropdown_count = await voice_dropdowns.count()
            
            self.logger.debug(f"[TTS] Found {dropdown_count} voice dropdown(s)")

            if dropdown_count == 0:
                self.logger.warning("[TTS] Voice dropdown not found")
                return False
            
            if voice_index >= dropdown_count:
                self.logger.warning(f"[TTS] Voice dropdown index {voice_index} out of range (total: {dropdown_count})")
                return False

            voice_dropdown = voice_dropdowns.nth(voice_index)
            
            # Scroll the dropdown into view (important for Speaker 2 which may be off-screen)
            await voice_dropdown.scroll_into_view_if_needed(timeout=5000)
            await asyncio.sleep(0.3)
            
            await voice_dropdown.click(timeout=5000)
            await asyncio.sleep(0.5)

            # Find and click the voice option in the opened listbox
            # The voice name is in a div with class "name" inside each mat-option
            # e.g., <div class="name">Puck</div>
            voice_option = self.page.locator(
                f'mat-option:has(.name:text-is("{voice}")), '
                f'mat-option .name:text-is("{voice}"), '
                f'[role="option"]:has(.name:text-is("{voice}")), '
                f'.cdk-overlay-container mat-option:has-text("{voice}")'
            ).first

            if await voice_option.count() > 0:
                await voice_option.click(timeout=5000)
                await asyncio.sleep(0.3)
                self.logger.info(f"[TTS] Selected voice: {voice}")
                return True
            else:
                self.logger.warning(f"[TTS] Voice option '{voice}' not found")
                # Press Escape to close dropdown
                await self.page.keyboard.press("Escape")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] Failed to select voice: {e}")
            try:
                await self.page.keyboard.press("Escape")
            except Exception:
                pass
            return False

    async def fill_multi_speaker_dialogs(
        self, speakers: List[TTSSpeaker], style_instructions: Optional[str] = None
    ) -> bool:
        """
        Fill in multiple speaker dialogs for multi-speaker mode.
        
        In multi-speaker mode, we replace the content in the "Raw structure" textarea
        on the left panel. The format is:
        
        [style instructions]
        Speaker 1: [text for speaker 1]
        Speaker 2: [text for speaker 2]
        Speaker 1: [next text for speaker 1]
        ...
        
        Speaker turns alternate between Speaker 1 and Speaker 2.
        The Script Builder on the right will automatically parse this format.
        We then select the appropriate voices from the right panel.

        Args:
            speakers: List of speaker objects with text and voice
            style_instructions: Optional style instructions

        Returns:
            True if successful
        """
        set_request_id(self.req_id)
        self.logger.info(f"[TTS] Filling multi-speaker dialog ({len(speakers)} turns)")

        try:
            # Build the raw structure text
            # Format: style instructions on first line(s), then Speaker N: text
            # Speakers alternate: 1, 2, 1, 2, ...
            raw_text_parts: List[str] = []
            
            # Add style instructions if provided
            if style_instructions:
                raw_text_parts.append(style_instructions)
            
            # Track voices for Speaker 1 and Speaker 2
            # First occurrence of each speaker determines their voice
            speaker_voices: Dict[int, Optional[str]] = {1: None, 2: None}
            
            # Add speaker turns - alternate between Speaker 1 and Speaker 2
            for i, speaker in enumerate(speakers):
                # Alternate: 0->1, 1->2, 2->1, 3->2, etc.
                speaker_num = (i % 2) + 1
                raw_text_parts.append(f"Speaker {speaker_num}: {speaker.text}")
                
                # Record voice for this speaker (first occurrence wins)
                if speaker.voice and speaker_voices[speaker_num] is None:
                    speaker_voices[speaker_num] = speaker.voice
            
            raw_text = "\n".join(raw_text_parts)
            self.logger.debug(f"[TTS] Raw structure text: {raw_text[:200]}...")
            self.logger.debug(f"[TTS] Speaker voices: {speaker_voices}")
            
            # Find the Raw structure textarea and fill it
            # Try multiple selectors to find the textarea
            raw_textarea = None
            
            # Try the defined selector first
            raw_textarea_locator = self.page.locator(TTS_RAW_STRUCTURE_SELECTOR).first
            if await raw_textarea_locator.count() > 0:
                raw_textarea = raw_textarea_locator
            else:
                # Fallback: find textarea in the left panel (Raw structure section)
                # The left panel has "Raw structure" header
                self.logger.debug("[TTS] Trying fallback Raw structure selector")
                
                # Try to find any textarea that might be the raw structure input
                all_textareas = self.page.locator('textarea')
                textarea_count = await all_textareas.count()
                self.logger.debug(f"[TTS] Found {textarea_count} textarea element(s) on page")
                
                # The raw structure textarea typically contains the default text
                for i in range(textarea_count):
                    ta = all_textareas.nth(i)
                    try:
                        value = await ta.input_value(timeout=1000)
                        # Check if it contains the default multi-speaker text pattern
                        if "Speaker 1:" in value or "Speaker 2:" in value:
                            raw_textarea = ta
                            self.logger.debug(f"[TTS] Found Raw structure textarea (index {i})")
                            break
                    except Exception:
                        continue
            
            if raw_textarea is None:
                self.logger.error("[TTS] Raw structure textarea not found")
                return False
            
            # Clear and fill the textarea with our formatted content
            await raw_textarea.click(timeout=5000)
            await asyncio.sleep(0.2)
            
            # Select all and replace
            await self.page.keyboard.press("Control+a")
            await asyncio.sleep(0.1)
            
            await raw_textarea.fill(raw_text, timeout=5000)
            await asyncio.sleep(0.5)
            
            self.logger.info("[TTS] Raw structure filled")
            
            # Now select voices for Speaker 1 and Speaker 2 from the right panel
            # Default voices are Zephyr (Speaker 1) and Puck (Speaker 2)
            default_voices = {1: "Zephyr", 2: "Puck"}
            
            for speaker_num in [1, 2]:
                voice = speaker_voices.get(speaker_num)
                if voice:
                    default_voice = default_voices[speaker_num]
                    if voice.lower() != default_voice.lower():
                        self.logger.info(f"[TTS] Speaker {speaker_num}: changing voice from {default_voice} to {voice}")
                        # voice_index: 0 for Speaker 1, 1 for Speaker 2
                        await self.select_voice(voice, voice_index=speaker_num - 1)
                        await asyncio.sleep(0.3)
                    else:
                        self.logger.debug(f"[TTS] Speaker {speaker_num}: using default voice {default_voice}")
            
            self.logger.info("[TTS] Multi-speaker dialog filled")
            return True

        except Exception as e:
            self.logger.error(f"[TTS] Failed to fill multi-speaker dialog: {e}")
            return False

    async def run_tts_generation(
        self, check_client_disconnected: Callable[[str], bool]
    ) -> bool:
        """
        Click the Run button to start TTS generation.

        Args:
            check_client_disconnected: Function to check if client disconnected

        Returns:
            True if generation started successfully
        """
        set_request_id(self.req_id)
        self.logger.info("[TTS] Starting TTS generation")

        try:
            await self._check_disconnect(check_client_disconnected, "Before TTS Run")

            run_button = self.page.locator(TTS_RUN_BUTTON_SELECTOR).first

            if await run_button.count() == 0:
                self.logger.error("[TTS] Run button not found")
                return False

            # Wait for button to be enabled
            try:
                await expect_async(run_button).to_be_enabled(timeout=10000)
            except Exception:
                self.logger.warning("[TTS] Run button may be disabled, attempting click anyway")

            await run_button.click(timeout=5000)
            self.logger.info("[TTS] TTS generation started")

            await self._check_disconnect(check_client_disconnected, "After TTS Run")
            return True

        except Exception as e:
            self.logger.error(f"[TTS] Failed to start TTS generation: {e}")
            return False

    async def wait_for_audio_ready(
        self,
        check_client_disconnected: Callable[[str], bool],
        timeout_ms: int = 600000,
    ) -> bool:
        """
        Wait for the audio to be generated and ready.

        Args:
            check_client_disconnected: Function to check if client disconnected
            timeout_ms: Maximum time to wait in milliseconds

        Returns:
            True if audio is ready
        """
        set_request_id(self.req_id)
        self.logger.info("[TTS] Waiting for audio generation")

        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_ms / 1000

        try:
            while True:
                await self._check_disconnect(
                    check_client_disconnected, "Waiting for TTS Audio"
                )

                stream_error_code, stream_error_message = self._poll_stream_proxy_error()
                if stream_error_code:
                    raise TTSGenerationError(
                        code=stream_error_code,
                        message=stream_error_message or "Upstream proxy error detected",
                    )

                error_code, error_message = await self._detect_generation_error()
                if error_code:
                    raise TTSGenerationError(
                        code=error_code,
                        message=error_message or "TTS generation failed",
                    )

                # Check for audio player or download button
                audio_player = self.page.locator(TTS_AUDIO_PLAYER_SELECTOR)
                download_button = self.page.locator(TTS_DOWNLOAD_BUTTON_SELECTOR)

                if await audio_player.count() > 0:
                    # Check if audio has a source
                    audio_src = await audio_player.first.get_attribute("src")
                    if audio_src:
                        self.logger.info("[TTS] Audio player is ready")
                        return True

                if await download_button.count() > 0:
                    if await download_button.first.is_visible(timeout=1000):
                        self.logger.info("[TTS] Download button is ready")
                        return True

                # Check for timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    self.logger.error("[TTS] Timed out waiting for audio")
                    raise TTSGenerationError(
                        code="tts_generation_timeout",
                        message="Audio generation timed out",
                    )

                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"[TTS] Waiting for audio failed: {e}")
            if isinstance(e, TTSGenerationError):
                raise
            return False

    def _poll_stream_proxy_error(self) -> Tuple[Optional[str], Optional[str]]:
        """Poll stream proxy queue for fail-fast upstream errors during TTS."""
        try:
            from server import STREAM_QUEUE

            if STREAM_QUEUE is None:
                return None, None

            while True:
                try:
                    item = STREAM_QUEUE.get_nowait()
                except (queue.Empty, asyncio.QueueEmpty):
                    return None, None

                if item is None:
                    continue

                payload: Optional[Dict[str, Any]] = None
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            payload = cast(Dict[str, Any], parsed)
                    except Exception:
                        continue
                elif isinstance(item, dict):
                    payload = cast(Dict[str, Any], item)

                if not payload or payload.get("error") is not True:
                    continue

                status_raw = payload.get("status", 0)
                if isinstance(status_raw, int):
                    status = status_raw
                elif isinstance(status_raw, str) and status_raw.isdigit():
                    status = int(status_raw)
                else:
                    status = 0
                message = str(payload.get("message", "Unknown upstream error"))

                if status == 429:
                    return "tts_too_many_requests", message
                if status == 400:
                    return "tts_upstream_bad_request", message
                if status >= 400:
                    return "tts_upstream_error", message

            return None, None
        except Exception as poll_err:
            self.logger.debug(f"[TTS] Stream proxy polling skipped due to: {poll_err}")
            return None, None

    async def _detect_generation_error(self) -> Tuple[Optional[str], Optional[str]]:
        """Detect explicit Too Many Requests/429 hints from the page text."""
        try:
            page_text = await self.page.evaluate(
                """
                () => {
                    const root = document.body;
                    if (!root) return '';
                    const text = root.innerText || '';
                    return text.slice(0, 25000);
                }
                """
            )
            if not page_text:
                return None, None

            lower_text = page_text.lower()

            # Keep this strict to avoid false positives from generic page text.
            has_explicit_too_many_requests = "too many requests" in lower_text
            has_429_request_phrase = "429" in lower_text and "request" in lower_text

            if has_explicit_too_many_requests or has_429_request_phrase:
                return (
                    "tts_too_many_requests",
                    "Upstream returned Too Many Requests (429)",
                )

            return None, None
        except Exception as detect_err:
            self.logger.debug(f"[TTS] Error detection skipped due to: {detect_err}")
            return None, None

    async def get_audio_data(self) -> Optional[bytes]:
        """
        Get the generated audio data.

        Returns:
            Audio data as bytes, or None if failed
        """
        set_request_id(self.req_id)
        self.logger.info("[TTS] Retrieving audio data")

        try:
            # Try to get audio from the audio player element
            audio_player = self.page.locator(TTS_AUDIO_PLAYER_SELECTOR).first

            if await audio_player.count() > 0:
                audio_src = await audio_player.get_attribute("src")

                if audio_src:
                    # If it's a blob URL, we need to fetch it via JavaScript
                    if audio_src.startswith("blob:"):
                        audio_data = await self._fetch_blob_audio(audio_src)
                        if audio_data:
                            return audio_data

                    # If it's a data URL
                    elif audio_src.startswith("data:"):
                        # Extract base64 data from data URL
                        parts = audio_src.split(",", 1)
                        if len(parts) == 2:
                            audio_base64 = parts[1]
                            return base64.b64decode(audio_base64)

                    # If it's a regular URL, fetch it
                    else:
                        audio_data = await self._fetch_url_audio(audio_src)
                        if audio_data:
                            return audio_data

            # Try download button as fallback
            download_button = self.page.locator(TTS_DOWNLOAD_BUTTON_SELECTOR).first

            if await download_button.count() > 0:
                download_url = await download_button.get_attribute("href")
                if download_url:
                    audio_data = await self._fetch_url_audio(download_url)
                    if audio_data:
                        return audio_data

            self.logger.error("[TTS] Unable to retrieve audio data")
            return None

        except Exception as e:
            self.logger.error(f"[TTS] Failed to retrieve audio data: {e}")
            return None

    async def _fetch_blob_audio(self, blob_url: str) -> Optional[bytes]:
        """Fetch audio data from a blob URL using JavaScript."""
        try:
            # Use JavaScript to fetch the blob and convert to base64
            audio_base64 = await self.page.evaluate(
                """
                async (blobUrl) => {
                    try {
                        const response = await fetch(blobUrl);
                        const blob = await response.blob();
                        return new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64 = reader.result.split(',')[1];
                                resolve(base64);
                            };
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        });
                    } catch (e) {
                        return null;
                    }
                }
                """,
                blob_url,
            )

            if audio_base64:
                return base64.b64decode(audio_base64)
            return None

        except Exception as e:
            self.logger.error(f"[TTS] Failed to fetch blob audio: {e}")
            return None

    async def _fetch_url_audio(self, url: str) -> Optional[bytes]:
        """Fetch audio data from a URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
            return None

        except Exception as e:
            self.logger.error(f"[TTS] Failed to fetch URL audio: {e}")
            return None

    async def process_tts_request(
        self,
        request: TTSRequest,
        check_client_disconnected: Callable[[str], bool],
    ) -> Optional[bytes]:
        """
        Process a complete TTS request.

        Args:
            request: The TTS request object
            check_client_disconnected: Function to check if client disconnected

        Returns:
            Audio data as bytes, or None if failed
        """
        set_request_id(self.req_id)
        self.logger.info("[TTS] Processing TTS request")

        try:
            # 1. Navigate to TTS page
            if not await self.navigate_to_tts_page(request.model):
                raise TTSGenerationError(
                    code="tts_navigation_failed",
                    message="Failed to navigate to TTS page",
                )

            await self._check_disconnect(
                check_client_disconnected, "After TTS Page Navigation"
            )

            # 2. Determine mode (single vs multi-speaker)
            is_multi_speaker = request.speakers is not None and len(request.speakers) > 0

            # 3. Set mode if needed
            await self.set_tts_mode(is_multi_speaker)

            await self._check_disconnect(check_client_disconnected, "After Mode Set")

            # 4 & 5. Fill in content (handling differs by mode)
            if is_multi_speaker and request.speakers:
                # Multi-speaker mode: fill raw structure textarea with formatted text
                # This includes both style instructions and speaker turns
                await self.fill_multi_speaker_dialogs(
                    request.speakers, 
                    style_instructions=request.style_instructions
                )
            else:
                # Single-speaker mode
                # Fill style instructions first
                if request.style_instructions:
                    await self.fill_style_instructions(request.style_instructions)
                
                # Fill text input
                if request.input:
                    await self.fill_text_input(request.input)

                # Select voice if different from default
                if request.voice:
                    await self.select_voice(request.voice)

            await self._check_disconnect(
                check_client_disconnected, "After Content Fill"
            )

            # 6. Run TTS generation
            if not await self.run_tts_generation(check_client_disconnected):
                raise TTSGenerationError(
                    code="tts_run_failed",
                    message="Failed to start TTS generation",
                )

            # 7. Wait for audio to be ready
            if not await self.wait_for_audio_ready(check_client_disconnected):
                raise TTSGenerationError(
                    code="tts_generation_timeout",
                    message="Audio generation timed out",
                )

            # 8. Get audio data
            audio_data = await self.get_audio_data()

            if audio_data:
                self.logger.info(f"[TTS] Audio generated successfully ({len(audio_data)} bytes)")
                return audio_data
            else:
                raise TTSGenerationError(
                    code="tts_audio_retrieval_failed",
                    message="Failed to retrieve audio data",
                )

        except Exception as e:
            self.logger.error(f"[TTS] TTS request processing failed: {e}")
            raise
