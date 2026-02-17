"""
TTS Page Controller Module

Handles browser interactions for TTS (Text-to-Speech) functionality in AI Studio.
Manages the TTS-specific UI elements for both single-speaker and multi-speaker modes.
"""

import asyncio
import base64
import logging
from typing import Callable, List, Optional

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
        self.logger.info(f"[TTS] 导航到TTS页面 (Model: {model_id})")

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
                    self.logger.info("[TTS] TTS页面加载成功")
                    return True
                else:
                    self.logger.warning("[TTS] TTS页面元素未找到，尝试备用检测")
                    # Check for run button as fallback
                    run_button = self.page.locator(TTS_RUN_BUTTON_SELECTOR)
                    if await run_button.count() > 0:
                        return True

            except Exception as e:
                self.logger.warning(f"[TTS] 页面验证失败: {e}")

            return False

        except Exception as e:
            self.logger.error(f"[TTS] 导航到TTS页面失败: {e}")
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
        self.logger.info(f"[TTS] 设置TTS模式: {mode_name}")

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
                self.logger.info(f"[TTS] {mode_name}模式已启用")
                return True
            else:
                self.logger.warning(f"[TTS] {mode_name}模式按钮未找到")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] 设置模式失败: {e}")
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
        self.logger.debug(f"[TTS] 填充风格指令 ({len(style)} chars)")

        try:
            style_input = self.page.locator(TTS_STYLE_INSTRUCTIONS_SELECTOR).first

            if await style_input.count() > 0:
                await style_input.fill(style, timeout=5000)
                self.logger.debug("[TTS] 风格指令已填充")
                return True
            else:
                self.logger.warning("[TTS] 风格指令输入框未找到")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] 填充风格指令失败: {e}")
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
        self.logger.debug(f"[TTS] 填充文本内容 ({len(text)} chars)")

        try:
            text_input = self.page.locator(TTS_TEXT_INPUT_SELECTOR).first

            if await text_input.count() > 0:
                await text_input.fill(text, timeout=5000)
                self.logger.debug("[TTS] 文本内容已填充")
                return True
            else:
                self.logger.warning("[TTS] 文本输入框未找到")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] 填充文本内容失败: {e}")
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
        self.logger.info(f"[TTS] 选择语音: {voice} (index: {voice_index})")

        try:
            # Find voice dropdowns (mat-select elements)
            voice_dropdowns = self.page.locator(TTS_VOICE_DROPDOWN_SELECTOR)
            dropdown_count = await voice_dropdowns.count()
            
            self.logger.debug(f"[TTS] 找到 {dropdown_count} 个语音下拉框")

            if dropdown_count == 0:
                self.logger.warning("[TTS] 语音下拉框未找到")
                return False
            
            if voice_index >= dropdown_count:
                self.logger.warning(f"[TTS] 语音下拉框索引 {voice_index} 超出范围 (共 {dropdown_count} 个)")
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
                self.logger.info(f"[TTS] 已选择语音: {voice}")
                return True
            else:
                self.logger.warning(f"[TTS] 语音选项 '{voice}' 未找到")
                # Press Escape to close dropdown
                await self.page.keyboard.press("Escape")
                return False

        except Exception as e:
            self.logger.error(f"[TTS] 选择语音失败: {e}")
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
        self.logger.info(f"[TTS] 填充多说话人对话 ({len(speakers)} turns)")

        try:
            # Build the raw structure text
            # Format: style instructions on first line(s), then Speaker N: text
            # Speakers alternate: 1, 2, 1, 2, ...
            raw_text_parts = []
            
            # Add style instructions if provided
            if style_instructions:
                raw_text_parts.append(style_instructions)
            
            # Track voices for Speaker 1 and Speaker 2
            # First occurrence of each speaker determines their voice
            speaker_voices = {1: None, 2: None}
            
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
                self.logger.debug("[TTS] 尝试备用Raw structure选择器")
                
                # Try to find any textarea that might be the raw structure input
                all_textareas = self.page.locator('textarea')
                textarea_count = await all_textareas.count()
                self.logger.debug(f"[TTS] 页面上共有 {textarea_count} 个textarea")
                
                # The raw structure textarea typically contains the default text
                for i in range(textarea_count):
                    ta = all_textareas.nth(i)
                    try:
                        value = await ta.input_value(timeout=1000)
                        # Check if it contains the default multi-speaker text pattern
                        if "Speaker 1:" in value or "Speaker 2:" in value:
                            raw_textarea = ta
                            self.logger.debug(f"[TTS] 找到Raw structure textarea (index {i})")
                            break
                    except Exception:
                        continue
            
            if raw_textarea is None:
                self.logger.error("[TTS] Raw structure textarea未找到")
                return False
            
            # Clear and fill the textarea with our formatted content
            await raw_textarea.click(timeout=5000)
            await asyncio.sleep(0.2)
            
            # Select all and replace
            await self.page.keyboard.press("Control+a")
            await asyncio.sleep(0.1)
            
            await raw_textarea.fill(raw_text, timeout=5000)
            await asyncio.sleep(0.5)
            
            self.logger.info("[TTS] Raw structure已填充")
            
            # Now select voices for Speaker 1 and Speaker 2 from the right panel
            # Default voices are Zephyr (Speaker 1) and Puck (Speaker 2)
            default_voices = {1: "Zephyr", 2: "Puck"}
            
            for speaker_num in [1, 2]:
                voice = speaker_voices.get(speaker_num)
                if voice:
                    default_voice = default_voices[speaker_num]
                    if voice.lower() != default_voice.lower():
                        self.logger.info(f"[TTS] Speaker {speaker_num}: 更改语音从 {default_voice} 到 {voice}")
                        # voice_index: 0 for Speaker 1, 1 for Speaker 2
                        await self.select_voice(voice, voice_index=speaker_num - 1)
                        await asyncio.sleep(0.3)
                    else:
                        self.logger.debug(f"[TTS] Speaker {speaker_num}: 使用默认语音 {default_voice}")
            
            self.logger.info("[TTS] 多说话人对话已填充完成")
            return True

        except Exception as e:
            self.logger.error(f"[TTS] 填充多说话人对话失败: {e}")
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
        self.logger.info("[TTS] 开始TTS生成")

        try:
            await self._check_disconnect(check_client_disconnected, "Before TTS Run")

            run_button = self.page.locator(TTS_RUN_BUTTON_SELECTOR).first

            if await run_button.count() == 0:
                self.logger.error("[TTS] Run按钮未找到")
                return False

            # Wait for button to be enabled
            try:
                await expect_async(run_button).to_be_enabled(timeout=10000)
            except Exception:
                self.logger.warning("[TTS] Run按钮可能未启用，仍尝试点击")

            await run_button.click(timeout=5000)
            self.logger.info("[TTS] TTS生成已开始")

            await self._check_disconnect(check_client_disconnected, "After TTS Run")
            return True

        except Exception as e:
            self.logger.error(f"[TTS] 启动TTS生成失败: {e}")
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
        self.logger.info("[TTS] 等待音频生成完成")

        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_ms / 1000

        try:
            while True:
                await self._check_disconnect(
                    check_client_disconnected, "Waiting for TTS Audio"
                )

                # Check for audio player or download button
                audio_player = self.page.locator(TTS_AUDIO_PLAYER_SELECTOR)
                download_button = self.page.locator(TTS_DOWNLOAD_BUTTON_SELECTOR)

                if await audio_player.count() > 0:
                    # Check if audio has a source
                    audio_src = await audio_player.first.get_attribute("src")
                    if audio_src:
                        self.logger.info("[TTS] 音频播放器已就绪")
                        return True

                if await download_button.count() > 0:
                    if await download_button.first.is_visible(timeout=1000):
                        self.logger.info("[TTS] 下载按钮已就绪")
                        return True

                # Check for timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    self.logger.error("[TTS] 等待音频超时")
                    return False

                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"[TTS] 等待音频失败: {e}")
            return False

    async def get_audio_data(self) -> Optional[bytes]:
        """
        Get the generated audio data.

        Returns:
            Audio data as bytes, or None if failed
        """
        set_request_id(self.req_id)
        self.logger.info("[TTS] 获取音频数据")

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

            self.logger.error("[TTS] 无法获取音频数据")
            return None

        except Exception as e:
            self.logger.error(f"[TTS] 获取音频数据失败: {e}")
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
            self.logger.error(f"[TTS] Blob音频获取失败: {e}")
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
            self.logger.error(f"[TTS] URL音频获取失败: {e}")
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
        self.logger.info("[TTS] 处理TTS请求")

        try:
            # 1. Navigate to TTS page
            if not await self.navigate_to_tts_page(request.model):
                raise Exception("Failed to navigate to TTS page")

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
                raise Exception("Failed to start TTS generation")

            # 7. Wait for audio to be ready
            if not await self.wait_for_audio_ready(check_client_disconnected):
                raise Exception("Audio generation timed out")

            # 8. Get audio data
            audio_data = await self.get_audio_data()

            if audio_data:
                self.logger.info(f"[TTS] 音频生成成功 ({len(audio_data)} bytes)")
                return audio_data
            else:
                raise Exception("Failed to retrieve audio data")

        except Exception as e:
            self.logger.error(f"[TTS] TTS请求处理失败: {e}")
            raise
